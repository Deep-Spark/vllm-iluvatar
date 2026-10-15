# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash vision variant on Iluvatar.

Corresponds to: vllm/models/deepseek_v41/nvidia/vl_model.py

Thin multimodal wrapper around the Iluvatar text model
(``IluDeepseekV41ForCausalLM``), which is held as ``language_model``:

- the ViT tower, the aligner and the three learned delimiter vectors are the
  community modules (``vllm.models.deepseek_v4.common.vision``); nothing in
  them is Iluvatar-specific except the ViT attention, which is an OOT
  ``MMEncoderAttention`` override picked up by the community layer.
- merged embeddings enter the text model through ``inputs_embeds``, i.e.
  before its hyper-connection stream expansion. Raw ``input_ids`` keep
  flowing in as well (``requires_raw_input_tokens``) because the MoE router
  and engram both have to recognize image-span positions.

Encoder CUDA graphs are deliberately not implemented (no
``SupportsEncoderCudaGraph``): under PD disaggregation the tower only ever
runs on the prefill node, where the batch shapes are not graph-friendly.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Annotated

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    StageMissingLayer,
    WeightsMapper,
    maybe_prefix,
    no_init_weights,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.models.deepseek_v4.common.vision import (
    DeepseekV4Aligner,
    DeepseekV4ViT,
    run_dp_sharded_vision_tower,
)
from vllm.models.deepseek_v41.common.mm_preprocess import (
    IMAGE,
    IMAGE_END,
    IMAGE_NEW_LINE,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLMultiModalProcessor,
    DeepseekV4VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from vllm_iluvatar.models.deepseek_v4.fp8_model import _make_fp8_weights_mapper
from vllm_iluvatar.models.deepseek_v41.model import IluDeepseekV41ForCausalLM

logger = init_logger(f"vllm.{__name__}")

# The attribute the text model is registered under, i.e. the prefix the outer
# mapper puts on every text-checkpoint name.
_LM_PREFIX = "language_model."


class DeepseekV4VLImagePixelInputs(TensorSchema):
    """ViT patch inputs for one batched set of images.

    Dimensions:
        - np: Total ViT patches across images (sum of n_vit_h * n_vit_w)
        - c: Number of image channels (3)
        - p: ViT patch size
        - ni: Number of images
        - ns: Total image-span positions (sum of n_llm_h * (n_llm_w + 1) + 2)
    """

    patches: Annotated[
        torch.Tensor, TensorShape("np", 3, "p", "p", dynamic_dims={"np"})
    ]
    # [n_vit_h, n_vit_w] per image
    vit_grid: Annotated[torch.Tensor, TensorShape("ni", 2)]
    # [n_llm_h, n_llm_w] per image
    llm_grid: Annotated[torch.Tensor, TensorShape("ni", 2)]
    types: Annotated[torch.Tensor, TensorShape("ns", dynamic_dims={"ns"})]


def _make_v41_vl_weights_mapper() -> WeightsMapper:
    """Text-checkpoint mapping rules re-rooted under ``language_model.``.

    Same construction as community
    ``_make_deepseek_v4_vl_weights_mapper`` (``nvidia/vl_model.py``), on top
    of the Iluvatar text rules instead of the community ones. ``vision.*``,
    ``aligner.*`` and ``image_*`` need no rule: the checkpoint already names
    them the way this wrapper does.
    """
    base = _make_fp8_weights_mapper()
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": f"{_LM_PREFIX}model.layers.",
            "embed.": f"{_LM_PREFIX}model.embed.",
            "norm.": f"{_LM_PREFIX}model.norm.",
            "hc_head": f"{_LM_PREFIX}model.hc_head",
            "mtp.": f"{_LM_PREFIX}model.mtp.",
        },
        orig_to_new_regex=base.orig_to_new_regex,
        orig_to_new_suffix={
            "head.weight": f"{_LM_PREFIX}lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            # The DSpark draft heads are not supported for the vision variant;
            # drop their weights (community does the same).
            "mtp.": None,
        },
    )


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VLMultiModalProcessor,
    info=DeepseekV4VLProcessingInfo,
    dummy_inputs=DeepseekV4VLDummyInputsBuilder,
)
class IluDeepseekV41VLForCausalLM(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsEagle3,
):
    """Multimodal entry point for DeepSeek-V4.1 checkpoints with a ViT."""

    supports_encoder_tp_data = True

    # The MoE router and engram need raw token ids to detect image-span
    # positions (all carrying image_token_id, see community
    # ``common/mm_preprocess.py``).
    requires_raw_input_tokens = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return IMAGE_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.config = config
        self.multimodal_config = model_config.multimodal_config
        assert self.multimodal_config is not None

        # The tower is always built; _mark_tower_model stubs it out
        # (StageMissingLayer, weights skipped) when the image limit is 0.
        with (
            self._mark_tower_model(vllm_config, {"image"}),
            self._tower_on_first_pp_stage_only(vllm_config),
        ):
            self.use_data_parallel = is_vit_use_data_parallel(config.vision_n_heads)
            self.vision = DeepseekV4ViT(config)
            self.aligner = DeepseekV4Aligner(config)
            self.image_start = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.image_end = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.image_newline = nn.Parameter(
                torch.empty(config.hidden_size, dtype=torch.float32)
            )
            self.vision.to(dtype=model_config.dtype)
            self.aligner.to(dtype=model_config.dtype)

        with self._mark_language_model(vllm_config):
            self.language_model = IluDeepseekV41ForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        # ``load_weights`` resolves HF names into this wrapper's namespace up
        # front, so the text model's own mapper must be a no-op: its suffix
        # rules are not idempotent ("lm_head.weight".endswith("head.weight")
        # would re-fire and produce "lm_lm_head.weight").
        self.language_model.hf_to_vllm_mapper = WeightsMapper()
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.language_model.make_empty_intermediate_tensors
        )
        self.hf_to_vllm_mapper = _make_v41_vl_weights_mapper()

    @contextmanager
    def _tower_on_first_pp_stage_only(self, vllm_config: VllmConfig) -> Iterator[None]:
        """Stub the tower on every pipeline stage except the first.

        Only the first stage ever runs the encoder: the runner creates its
        ``EncoderCache`` and calls ``prepare_inputs_embeds`` under
        ``is_first_pp_rank``. Community nevertheless builds and loads the tower
        on every stage, which at TP2/PP8 pins ~0.46 GiB of idle ViT weights per
        card that would otherwise be KV cache. This is the mechanism
        ``_mark_tower_model`` already uses for a zero image limit, so the
        weights are dropped the same way (``AutoWeightsLoader`` skips a
        ``StageMissingLayer``) and the delimiter parameters stay on meta.
        """
        mm_config = vllm_config.model_config.multimodal_config
        already_stubbed = (
            mm_config is not None and mm_config.get_limit_per_prompt("image") == 0
        )
        if already_stubbed or get_pp_group().is_first_rank:
            yield
            return
        with no_init_weights(self, lambda mod: StageMissingLayer("vision_tower", mod)):
            yield

    # ---- image encoding ---------------------------------------------------

    def _encode_image(
        self,
        patches: torch.Tensor,
        n_vit_h: int,
        n_vit_w: int,
    ) -> torch.Tensor:
        # Aligner rows in reading order, one per IMAGE slot.
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def _build_image_span(
        self, image_embeds: torch.Tensor, types: torch.Tensor
    ) -> torch.Tensor:
        """Full image span: aligner rows at the IMAGE slots, the learned
        delimiter vectors at IMAGE_START/IMAGE_NEW_LINE/IMAGE_END."""
        types = types.to(image_embeds.device)
        span = image_embeds.new_empty(types.numel(), image_embeds.shape[-1])
        dtype = image_embeds.dtype
        span[types == IMAGE_START] = self.image_start.to(dtype)
        span[types == IMAGE_END] = self.image_end.to(dtype)
        span[types == IMAGE_NEW_LINE] = self.image_newline.to(dtype)
        span[types == IMAGE] = image_embeds
        return span

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> DeepseekV4VLImagePixelInputs | None:
        patches = kwargs.pop("patches", None)
        if patches is None:
            return None
        return DeepseekV4VLImagePixelInputs(
            patches=patches,
            vit_grid=kwargs.pop("vit_grid"),
            llm_grid=kwargs.pop("llm_grid"),
            types=kwargs.pop("types"),
            resolve_bindings={"p": self.config.vision_patch_size},
        )

    def _process_image_input(
        self,
        image_input: DeepseekV4VLImagePixelInputs,
    ) -> tuple[torch.Tensor, ...]:
        patches = image_input.patches.to(self.aligner.w1.weight.dtype)
        vit_grid = image_input.vit_grid.tolist()

        image_embeds_list: list[torch.Tensor]
        if self.use_data_parallel and get_tensor_model_parallel_world_size() > 1:
            # Data-parallel ViT: shard images across TP ranks and all-gather
            # the per-image embeddings (weights are replicated on every rank).
            image_embeds_list = run_dp_sharded_vision_tower(
                self.vision, self.aligner, patches, vit_grid
            )
        else:
            image_embeds_list = []
            vit_offset = 0
            for n_vit_h, n_vit_w in vit_grid:
                n_vit = n_vit_h * n_vit_w
                image_embeds_list.append(
                    self._encode_image(
                        patches[vit_offset : vit_offset + n_vit], n_vit_h, n_vit_w
                    )
                )
                vit_offset += n_vit

        embeds: list[torch.Tensor] = []
        span_offset = 0
        for image_embeds, (n_llm_h, n_llm_w) in zip(
            image_embeds_list, image_input.llm_grid.tolist(), strict=True
        ):
            span_len = n_llm_h * (n_llm_w + 1) + 2
            embeds.append(
                self._build_image_span(
                    image_embeds,
                    image_input.types[span_offset : span_offset + span_len],
                )
            )
            span_offset += span_len
        return tuple(embeds)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []
        return self._process_image_input(image_input)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        inputs_embeds = self.language_model.embed_input_ids(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    # ---- text model delegation --------------------------------------------

    @staticmethod
    def get_model_state_cls():
        return IluDeepseekV41ForCausalLM.get_model_state_cls()

    @property
    def token_lookback_depth(self) -> int:
        return self.language_model.token_lookback_depth

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        raw_input_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
            raw_input_ids=raw_input_ids,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_default_aux_hidden_state_layers()

    # ``get_mtp_target_hidden_states`` is absent on purpose, exactly as on the
    # text model: the runner probes for it with ``hasattr`` and V4.1 has no MTP
    # draft to hand a residual to.

    # ---- weights ----------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Split the mapped stream into the text model and the tower.

        The text model's ``load_weights`` is called directly rather than
        through ``AutoWeightsLoader``: it has to see the whole text checkpoint
        in one call (it finalizes fused expert weights, loads the engram tables
        from mmap and runs its load verifier at the end), and the community
        trick of sorting the stream to make ``language_model.`` one contiguous
        group would materialize all 96k checkpoint tensors at once.

        Draining that generator is what fills ``tower``; the tower tensors are
        the ~0.9 GiB of ``vision.*`` / ``aligner.*`` / ``image_*`` and are held
        until the text model is done.
        """
        tower: list[tuple[str, torch.Tensor]] = []

        def text_stream() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in self.hf_to_vllm_mapper.apply(weights):
                if name.startswith(_LM_PREFIX):
                    yield name[len(_LM_PREFIX) :], weight
                else:
                    tower.append((name, weight))

        loaded = {
            f"{_LM_PREFIX}{name}"
            for name in self.language_model.load_weights(text_stream())
        }
        logger.debug(
            "[IluDeepseekV41VLForCausalLM.load_weights] %d tower tensor(s) "
            "after %d text parameter(s)",
            len(tower),
            len(loaded),
        )
        return loaded | AutoWeightsLoader(self).load_weights(tower)


def register_model() -> None:
    from vllm import ModelRegistry

    # Both released arch names land here, not on the text model, following
    # community: upstream maps ``DeepseekV41ForCausalLM`` to its vision wrapper
    # as well and lets a text-only checkpoint degrade (``vision_n_layers == 0``
    # builds an empty tower, which ``_mark_tower_model`` then stubs out --
    # ``IluPlatform.check_and_update_config`` turns that stubbing on
    # automatically). The explicit ``...VLForCausalLM`` name is an alias, for
    # forcing the vision path via ``--hf-overrides``.
    for arch in (
        "DeepseekV41ForCausalLM",
        "IluDeepseekV41ForCausalLM",
        "IluDeepseekV41VLForCausalLM",
    ):
        ModelRegistry.register_model(
            arch,
            "vllm_iluvatar.models.deepseek_v41.vl_model:IluDeepseekV41VLForCausalLM",
        )

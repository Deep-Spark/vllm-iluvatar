# SPDX-License-Identifier: Apache-2.0
"""MLA attention and lightning indexer for HY V4 on Iluvatar.

Corresponds to: vllm/models/hy_v4/nvidia/attention.py (upstream #54160)

Reuses community `MLAAttention` rather than reimplementing the algorithm chain:
``IluPlatform`` already redirects ``AttentionBackendEnum.FLASHMLA_SPARSE`` to
``IluFlashMLASparseBackend``, so a community ``MLAAttention(use_sparse=True)``
lands on ixformer kernels with no model-side work. The HY V4 deltas are the
per-head learnable sink (bound through `.flashmla_sparse`) and the gated-MLA
output gate.

The indexer instead takes the plugin ``IluSparseAttnIndexer`` /
``IluDeepseekV32IndexerCache`` (as V3.2 and GLM DSA do). ixformer keeps the
indexer K cache in BF16 and quantizes nothing, so the community fp8 indexer
contract -- a per-128-block-quantized q with its scale folded into ``weights``,
and a uint8 cache carrying fp32 block scales -- does not apply here. It also
cannot: ``per_token_group_quant_fp8`` needs ``torch.ops._C.
per_token_group_fp8_quant``, which Iluvatar does not build.

Delta vs upstream: the HPC fused output gate (``hpc_gated_mla_gemm``) is
dropped. It needs the hpc ``.so`` plus sm100/sm103 and is off by default
(``VLLM_ENABLE_HPC_OPS=0``), so upstream's own default is the eager gate kept
below.
"""

from __future__ import annotations

from typing import cast

import regex as re
import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config, PretrainedConfig
from vllm.compilation.breakable_cudagraph import (
    eager_break_during_capture,
    is_breakable_cudagraph_enabled,
)
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.selector import get_attn_backend

from vllm_iluvatar.model_executor.layers.sparse_attn_indexer import (
    IluSparseAttnIndexer,
)
from vllm_iluvatar.models.deepseek_v32.attention import IluDeepseekV32IndexerCache

from .topk_index_cache import HYV4TopkIndexCache

logger = init_logger(__name__)

_SPARSE_LAYER_TYPES = ("sparse_attention", "sparse", "deepseek_sparse_attention")
_WEIGHT_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def compute_skip_topk_layers(config: PretrainedConfig) -> set[int]:
    """Return the backbone layers that reuse a previous layer's top-k indices.

    A "shared" indexer layer performs sparse attention with the indices computed
    by the closest preceding "full" indexer layer, so it does not build its own
    indexer and its checkpoint indexer weights must be skipped.

    Args:
        config: The model config.

    Returns:
        The set of layer indices that share another layer's top-k indices.

    Raises:
        ValueError: If ``indexer_types`` has the wrong length or an unknown
            entry, or if ``index_topk_freq`` is not a positive integer.
    """
    if not hasattr(config, "index_topk"):
        return set()

    num_hidden_layers = config.num_hidden_layers
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        if len(indexer_types) != num_hidden_layers:
            raise ValueError(
                "indexer_types must contain one entry per hidden layer: "
                f"expected {num_hidden_layers}, got {len(indexer_types)}."
            )
        invalid_types = sorted(set(indexer_types) - {"full", "shared"})
        if invalid_types:
            raise ValueError(
                f"indexer_types only supports 'full' and 'shared', got {invalid_types}."
            )
        return {
            layer_idx
            for layer_idx, indexer_type in enumerate(indexer_types)
            if indexer_type == "shared"
        }

    freq = getattr(config, "index_topk_freq", 1)
    if not isinstance(freq, int) or freq <= 0:
        raise ValueError(f"index_topk_freq must be a positive integer, got {freq!r}.")
    pattern = getattr(config, "index_topk_pattern", None)
    offset = getattr(config, "index_skip_topk_offset", 2)
    skip_layers: set[int] = set()
    for layer_idx in range(num_hidden_layers):
        if pattern is None:
            if max(layer_idx - offset + 1, 0) % freq != 0:
                skip_layers.add(layer_idx)
        elif 0 <= layer_idx < len(pattern) and pattern[layer_idx] == "S":
            skip_layers.add(layer_idx)
    return skip_layers


def is_skip_topk_indexer_weight(weight_name: str, skip_topk_layers: set[int]) -> bool:
    """Return whether an indexer weight belongs to a top-k sharing layer.

    Args:
        weight_name: Checkpoint weight name.
        skip_topk_layers: Result of `compute_skip_topk_layers`.

    Returns:
        True when the weight is an indexer weight of a layer that has no
        indexer module and therefore must be dropped.
    """
    if ".indexer." not in weight_name or not skip_topk_layers:
        return False
    match = _WEIGHT_LAYER_INDEX_RE.search(weight_name)
    return match is not None and int(match.group(1)) in skip_topk_layers


class Indexer(nn.Module):
    """Lightning indexer selecting the top-k tokens for sparse MLA."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

        # No tensor parallelism, just replicated.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # Quantized wk weights are upcast to BF16 while loading to keep the
        # fusion (see ``weight_utils.try_load_quantized_indexer_wk``).
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5
        self.n_head_scale = self.n_head**-0.5
        # Both scales are constants, so fold them once here: the product is
        # 1/sqrt(index_head_dim * index_n_heads), which is an exact power of two
        # for HY4 (1/64), so the single multiply in ``prepare_inputs`` is one
        # bf16 rounding fewer than applying the two factors in turn.
        self.weights_scale = self.softmax_scale * self.n_head_scale

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        # BF16 indexer K cache, matching the other Iluvatar DSA models: the
        # ixformer indexer keeps K in the checkpoint dtype rather than the
        # community fp8 + per-128-block-scale packing, which is also why no q
        # quantization happens below.
        assert cache_config is not None, "HYV4 indexer requires cache_config"
        self.k_cache = IluDeepseekV32IndexerCache(
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = IluSparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        hidden_states, q, k, weights = self.prepare_inputs(
            hidden_states, qr, positions, rotary_emb
        )
        return self.indexer_op(hidden_states, q, k, weights)

    def prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the query, key and per-head weights of the indexer."""
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        kw, _ = self.wk_weights_proj(hidden_states)
        k = self.k_norm(kw[:, : self.head_dim])

        # Checkpoint (PTM) layout: pe occupies the LAST rope_dim dims, and both
        # tensors already hold no_pe first, so RoPE can rotate the tail in
        # place. ``ops.rotary_embedding`` reads its strides from the tensor
        # descriptor, so the views below are rotated inside q / k themselves and
        # the two [num_tokens, head_dim] cats that used to reassemble them are
        # gone. q and k only have to agree with each other -- the indexer logits
        # kernel dots over the whole head_dim -- so the checkpoint order is kept.
        nope_dim = self.head_dim - self.rope_dim
        q_pe = q[..., nope_dim:]
        # ``k_pe`` is [num_tokens, 1, rope_dim] (MQA).
        k_pe = k[:, nope_dim:].unsqueeze(1)
        q_pe_out, k_pe_out = rotary_emb(positions, q_pe, k_pe)
        if q_pe_out is not q_pe:
            # The CustomOp fallback (``forward_native``) returns fresh tensors
            # instead of rotating in place, so copy them back.
            q_pe.copy_(q_pe_out.reshape(q_pe.shape))
            k_pe.copy_(k_pe_out.reshape(k_pe.shape))

        # The multiply doubles as the contiguous copy the indexer logits kernel
        # requires: ``weights`` is a [num_tokens, n_head] column slice of the
        # fused wk/weights_proj GEMM, so it is row-strided on its own.
        weights = kw[:, self.head_dim :] * self.weights_scale
        return hidden_states, q, k, weights


class HYV4MLAAttention(nn.Module):
    """Multi-head latent attention with optional sparse lightning indexer.

    Main reference: the DeepSeek-V2 paper and the FlashInfer implementation
    (https://arxiv.org/abs/2405.04434). HY V4 additionally supports an output
    gate (``gated_mla``) and a per-head learnable attention sink.

    The sink is applied by binding the sink-capable backend from
    `.flashmla_sparse`; if no backend on this platform can consume sinks, the
    weight is still loaded but the bias is disabled with a warning.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        layer_idx: int = 0,
        topk_index_cache: HYV4TopkIndexCache | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.layer_idx = layer_idx
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.layer_id = int(prefix.split(".")[-2])
        layer_types = getattr(config, "layer_types", None)
        requested_sparse = (
            hasattr(config, "index_topk")
            and layer_types is not None
            and self.layer_id < len(layer_types)
            and layer_types[self.layer_id] in _SPARSE_LAYER_TYPES
        )
        # Only actual sparse layers may share another layer's top-k indices.
        self.skip_topk = requested_sparse and self.layer_id in compute_skip_topk_layers(
            config
        )
        # The skip pattern only governs backbone layers. MTP/nextn layers
        # (layer_id >= num_hidden_layers) always build a full indexer: they
        # compute indices at draft step 0 and toggle at runtime.
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        is_mtp_layer = (
            num_hidden_layers is not None and self.layer_id >= num_hidden_layers
        )
        self.create_indexer = requested_sparse and (not self.skip_topk or is_mtp_layer)
        self.is_sparse = requested_sparse

        # Do not silently degrade sparse layers into dense attention. Probe the
        # sparse MLA backend directly and fail fast with the real error.
        kv_cache_dtype = cache_config.cache_dtype if cache_config else "auto"
        if self.is_sparse:
            try:
                get_attn_backend(
                    head_size=self.kv_lora_rank + self.qk_rope_head_dim,
                    dtype=torch.get_default_dtype(),
                    kv_cache_dtype=kv_cache_dtype,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=True,
                    num_heads=self.num_local_heads,
                )
            except Exception as exc:
                raise RuntimeError(
                    "HYV4 sparse attention was requested, but no valid sparse MLA "
                    "backend is available for current runtime/config. "
                    "Refusing to fall back to dense attention."
                ) from exc

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.fused_qkv_a_proj = None
        self.kv_a_proj_with_mqa = None
        if self.q_lora_rank is not None:
            # ``q_a_proj`` and ``kv_a_proj_with_mqa`` read the same
            # ``hidden_states`` and are both TP-replicated, so they run as one
            # GEMM. The checkpoint keeps them separate; `load_weights` merges
            # them through ``stacked_params_mapping``.
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        self.q_a_layernorm = None
        self.q_b_proj = None
        self.q_proj = None
        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        self.indexer_rope_emb: nn.Module | None
        self.indexer: Indexer | None
        if self.create_indexer:
            # The checkpoint stores indexer q_pe/k_pe in interleaved
            # (Megatron/PTM) layout, so the indexer must use interleaved RoPE
            # (is_neox_style=False) like the main attention path. Using NeoX
            # here loses the relative-position dependence and corrupts the DSA
            # top-k selection.
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )
            # The indexer projects its queries from the MLA q_lora activations,
            # so a sparse layer requires a query down-projection.
            assert q_lora_rank is not None, (
                "HYV4 sparse attention requires q_lora_rank to be set"
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

        self.gated_mla = bool(getattr(config, "gated_mla", False))
        self.linear_gate: ColumnParallelLinear | None
        if self.gated_mla:
            if config.gating_type == "headwise":
                self.gate_projection_size_per_head = 1
            elif config.gating_type == "elementwise":
                self.gate_projection_size_per_head = self.v_head_dim
            else:
                raise ValueError(f"Unknown gating type: {config.gating_type}")
            self.linear_gate = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.gate_projection_size_per_head,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_gate",
            )
        else:
            self.linear_gate = None
        self.prefix = prefix

        # Per-head learnable attention sink. Created BEFORE ``MLAAttention`` so
        # it can be forwarded as the ``sinks`` impl kwarg. The parameter always
        # holds the local TP shard.
        self.learnable_sink = bool(getattr(config, "learnable_sink", False))
        sinks = None
        sink_backend: type[AttentionBackend] | None = None
        if self.learnable_sink:
            sink_backend = self._resolve_sink_backend(kv_cache_dtype)
            enable_sink = sink_backend is not None
            self.learnable_sink_param = nn.Parameter(
                torch.empty(
                    self.num_local_heads,
                    # The kernels require fp32 sinks; the disabled path keeps
                    # the checkpoint dtype since the value is never consumed.
                    dtype=torch.float32 if enable_sink else torch.bfloat16,
                )
            )
            if enable_sink:
                sinks = self.learnable_sink_param
                self._force_sparse_mqa()

        extra_impl_args: dict[str, object] = {} if sinks is None else {"sinks": sinks}
        if self.is_sparse and topk_index_cache is not None:
            # A "shared" layer reads what the closest preceding full layer left
            # in ``topk_indices_buffer``, so it can also reuse everything that
            # layer derived from it (see ``topk_index_cache``).
            extra_impl_args["topk_index_cache"] = topk_index_cache
            extra_impl_args["reuses_topk_indices"] = self.skip_topk
        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=topk_indices_buffer,
            attn_backend=sink_backend,
            **extra_impl_args,
        )
        self.assert_topk_buffer_shared()

    @property
    def topk_indices_buffer(self) -> torch.Tensor | None:
        """The sparse top-k index buffer this layer reads from.

        The real consumer is ``mla_attn.impl``, an ``AttentionImpl`` rather than
        an ``nn.Module``, so a plain ``named_modules()`` walk cannot reach it.
        The MTP proposer rebinds the draft layers onto the target model's buffer
        through such a walk, so expose it here: without this the draft attention
        would keep reading its own buffer while the indices it needs are written
        to the target's.
        """
        return getattr(self.mla_attn.impl, "topk_indices_buffer", None)

    @topk_indices_buffer.setter
    def topk_indices_buffer(self, buffer: torch.Tensor) -> None:
        # Three objects hold this buffer and all three must move together: the
        # impl *reads* the top-k rows, while the one that *writes* them is the
        # indexer's opaque op. Rebinding only the impl would leave the draft
        # reading the target's rows while its own indexer fills a buffer nobody
        # looks at -- top-k would be silently wrong (no error, accept rate near
        # zero). ``assert_topk_buffer_shared`` checks the invariant.
        self.mla_attn.impl.topk_indices_buffer = buffer  # type: ignore[attr-defined]
        if self.indexer is not None:
            self.indexer.topk_indices_buffer = buffer
            self.indexer.indexer_op.topk_indices_buffer = buffer

    def assert_topk_buffer_shared(self) -> None:
        """Fail fast unless reader and writer share one top-k buffer.

        Cheap (three ``data_ptr()`` reads, no device sync) and only meaningful
        on a layer that owns an indexer: a shared-topk layer has no writer of
        its own by design.
        """
        if self.indexer is None:
            return
        impl_buffer = self.topk_indices_buffer
        if impl_buffer is None:
            return
        holders = {
            "mla_attn.impl": impl_buffer,
            "indexer": self.indexer.topk_indices_buffer,
            "indexer.indexer_op": self.indexer.indexer_op.topk_indices_buffer,
        }
        pointers = {
            name: None if tensor is None else tensor.data_ptr()
            for name, tensor in holders.items()
        }
        if len(set(pointers.values())) != 1:
            raise RuntimeError(
                f"HYV4 layer {self.layer_id} top-k index buffer is not shared "
                f"between its reader and writer: {pointers}. The sparse "
                "attention would read indices the indexer never wrote."
            )

    def _resolve_sink_backend(
        self, kv_cache_dtype: str
    ) -> type[AttentionBackend] | None:
        """Return an MLA backend that can apply this layer's learnable sink.

        The sink is part of the architecture, so a backend that cannot apply it
        changes the model's output. Resolution order:

        1. If the backend the selector would pick already advertises
           `supports_sink`, keep it -- this also honours an explicit
           ``--attention-backend`` choice.
        2. Otherwise fall back to the sink-capable Iluvatar sparse subclass in
           `.flashmla_sparse`, whose ixformer kernel accepts ``attn_sink``,
           provided it validates against the current runtime configuration.
        3. Otherwise give up on the bias rather than failing the load.

        Args:
            kv_cache_dtype: The layer's KV cache dtype string.

        Returns:
            The backend class to bind, or None when no sink-capable backend is
            available; the caller then loads the sink weight but disables the
            bias.
        """
        head_size = self.kv_lora_rank + self.qk_rope_head_dim
        dtype = torch.get_default_dtype()
        try:
            selected_cls = get_attn_backend(
                head_size=head_size,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                use_mla=True,
                use_sparse=self.is_sparse,
                num_heads=self.num_local_heads,
            )
        except Exception as exc:
            # Stringify before logging: warning_once dedupes on the arguments,
            # and a fresh exception object per layer would defeat it.
            logger.warning_once(
                "HYV4 failed to select an MLA backend for the learnable sink "
                "(%s); the sink parameter is loaded but the sink bias is "
                "disabled.",
                str(exc),
            )
            return None

        if selected_cls.supports_sink():
            return selected_cls

        from .flashmla_sparse import HYV4FlashMLASparseBackend

        # Mirror how the selector derives the configuration-dependent inputs so
        # this check accepts exactly what the backend would accept at runtime.
        capability = current_platform.get_device_capability()
        if capability is None:
            logger.warning_once(
                "HYV4 learnable sink is unavailable: the device compute "
                "capability is unknown. The sink parameter is loaded but the "
                "sink bias is disabled."
            )
            return None
        cache_config = get_current_vllm_config().cache_config
        block_size = (
            cache_config.block_size
            if cache_config is not None and cache_config.user_specified_block_size
            else None
        )
        invalid_reasons = HYV4FlashMLASparseBackend.validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=cast(CacheDType, kv_cache_dtype),
            block_size=block_size,
            use_mla=True,
            has_sink=True,
            use_sparse=self.is_sparse,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=capability,
            attn_type=AttentionType.DECODER,
        )
        if invalid_reasons:
            logger.warning_once(
                "HYV4 learnable sink is unavailable: the selected backend %s "
                "cannot apply sinks and the sink-capable Iluvatar sparse path "
                "is invalid here (%s). The sink parameter is loaded but the "
                "sink bias is disabled.",
                selected_cls.get_name(),
                ", ".join(invalid_reasons),
            )
            return None

        logger.info_once(
            "HYV4 learnable sink enabled: using the sink-capable Iluvatar "
            "sparse MLA impl instead of %s, which cannot apply sinks.",
            selected_cls.get_name(),
        )
        return HYV4FlashMLASparseBackend

    def _force_sparse_mqa(self) -> None:
        """Keep every token on the sink-capable sparse MQA path.

        ``_resolve_sink_backend`` only binds the backend that serves decode.
        ``MLAAttention`` additionally routes short prefills to a separate dense
        MLA prefill backend, and none of those accept ``attn_sink``, so prefill
        would silently drop the sink while decode applies it. Such a partially
        applied sink is not the trained architecture and corrupts the output, so
        opt out of the dense split instead.

        Prefills up to ``index_topk`` keep every token inside the sparse top-k,
        making the sparse path numerically equivalent to the dense one apart from
        also applying the sink.
        """
        attention_config = get_current_vllm_config().attention_config
        if attention_config.sparse_mla_force_mqa:
            return
        attention_config.sparse_mla_force_mqa = True
        logger.info_once(
            "HYV4 learnable sink enabled: forcing sparse MQA for prefill too, "
            "as the dense MLA prefill backends cannot apply sinks."
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.q_proj is not None
            assert self.kv_a_proj_with_mqa is not None
            q = self.q_proj(hidden_states)[0]
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        # Add a head dim of 1 to k_pe.
        k_pe = k_pe.unsqueeze(1)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        output_shape = (
            hidden_states.shape[0],
            self.num_local_heads * self.v_head_dim,
        )
        if is_breakable_cudagraph_enabled():
            # Breakable CUDA graphs replay the eager segment without a Python
            # return value, so it needs a caller-owned stable output buffer.
            attn_out = torch.empty(
                output_shape, dtype=hidden_states.dtype, device=hidden_states.device
            )
            self._indexer_and_attn(
                hidden_states, q_c, positions, q, kv_c_normed, k_pe, attn_out
            )
        else:
            # FULL_DECODE_ONLY (the HY4 path) captures the whole function and
            # does not need the breakable in-place contract. Return the
            # MLAAttention buffer directly, avoiding one [T,H*V] allocation and
            # one full copy per layer.
            if self.indexer is not None and self.is_sparse and not self.skip_topk:
                self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
            attn_out = self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=output_shape,
            )

        if self.gated_mla and self.linear_gate is not None:
            gate_score = self.linear_gate(hidden_states)[0]
            if self.config.gating_type == "headwise":
                gate_score = gate_score.unsqueeze(-1)
                attn_out = attn_out.reshape(*attn_out.shape[:-1], -1, self.v_head_dim)
                attn_out = attn_out * torch.sigmoid(gate_score)
                attn_out = attn_out.reshape(*attn_out.shape[:-2], -1)
            else:
                # HY4 is ``elementwise``, one gate per output element. Both
                # operands are this layer's own freshly allocated buffers (the
                # gate GEMM output and, since O2, the MLAAttention output), so
                # gating in place drops two [num_tokens, heads * v_head_dim]
                # allocations per layer without aliasing anything o_proj reads.
                attn_out = attn_out.mul_(gate_score.sigmoid_())

        out, _ = self.o_proj(attn_out)
        return out

    @eager_break_during_capture
    def _indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | None,
        positions: torch.Tensor,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, heads * v_head_dim], written in place
    ) -> None:
        """Run the lightning indexer and MLA attention in one eager segment.

        Both read per-batch attention metadata, so under the breakable cudagraph
        they must not be captured. Keeping them in a single break (instead of one
        break each) also means the attention inputs never have to survive a
        capture-segment boundary.
        """
        if self.indexer is not None and self.is_sparse and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
        out.copy_(
            self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=out.shape,
            )
        )

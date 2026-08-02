# SPDX-License-Identifier: Apache-2.0
"""T2 (attention part): parameter-name / shape reconciliation for the DS-V4-Flash
raw-FP8 attention skeleton (``IluDeepseekV4Fp8Attention``).

What is covered
  1. The parameter-name set of the three layer flavours (dense / ``cr=128`` /
     ``cr=4``) -- no extra, no missing entry.
     A dense layer has neither ``compressor`` nor ``indexer`` params; a
     ``cr=128`` layer has ``compressor`` but no ``indexer``; a ``cr=4`` layer
     has both.
  2. Every parameter shape and dtype matches the table, folded down to the mini
     config by the same formulas the TP8 table uses.
  3. ``attn_sink`` is ``[padded_heads]`` with ``padded_heads ==
     get_padded_num_q_heads(n_local_heads) == 64`` (``64 if n <= 64 else 128``;
     the mini config shrinks ``n_local_heads``, so this is 64 regardless of TP),
     dtype fp32, and every slot initialised to ``-inf``. Weight loading later
     overwrites only the first ``n_local_heads`` slots -- here we only pin
     construction-time shape + init.

Runtime notes
  * No kernel runs. Structural assertions build on ``meta``; the ``attn_sink``
    value assertion builds the dense flavour on real CPU memory (meta tensors
    carry no values), and the dense flavour pulls in neither compressor nor
    indexer.
  * ``meta`` is not airtight for the ``cr=4`` / ``cr=128`` flavours: community
    ``DeepseekCompressor.__init__`` passes an explicit
    ``device=current_platform.device_type`` (``vllm/models/deepseek_v4/
    compressor.py``, ``self.device`` set from the platform) when allocating
    ``ape`` and its state buffers. An explicit ``device=`` wins over the
    ambient ``torch.device("meta")`` context, so those tensors really are
    allocated on the accelerator. That is fine on the target container (it has
    devices) and costs a few KB; it does mean these two flavours cannot be
    collected on a device-less host. No assertion depends on the placement --
    only shape / dtype / name are checked -- so nothing here is skipped for it.
  * A TP1 process group is needed (parallel linear layers read the TP world size
    even on ``meta``); it is created over gloo, see ``_tp1_distributed_env``.
  * Expected names/shapes below are hand-transcribed and never read back out
    of the implementation.
  * ``make_mini_vllm_config`` / ``build_mini_fp8_attention`` are reused by other
    2.3 and 3.5, hence the module-level (importable) definitions. They live here
    rather than in a ``conftest.py`` because ``tests/`` is already on
    ``sys.path`` and a second ``conftest`` module in this directory would shadow
    ``from conftest import ...`` used elsewhere in the suite.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
import inspect
import re
import textwrap
from typing import Any

import pytest
import torch

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
BF16 = torch.bfloat16
FP32 = torch.float32

TODO_RE = re.compile(r"TODO\(ilu_fp8_[A-E]\d+\)")

# --------------------------------------------------------------------------- #
# mini config
# --------------------------------------------------------------------------- #
# Only hidden / heads / experts / layers / vocab are shrunk. Every structural
# field of the real DeepSeek-V4-Flash config is kept verbatim, because the
# community sub-modules key their layout off them:
#   * head_dim=512 and index_head_dim=128  -- DeepseekCompressor only accepts
#     these two head dims (compressor.py raises otherwise)
#   * qk_rope_head_dim / q_lora_rank / o_lora_rank / o_groups -- drive the
#     fused_wqa_wkv, wq_b, wo_a, wo_b shapes
#   * index_n_heads -- drives indexer.wq_b / weights_proj
#   * num_hash_layers -- gate flavour split (used by tasks 3.x)
#   * compress_ratios keeps all three classes: dense(0) / 4 / 128, plus the
#     trailing MTP entry (len == num_hidden_layers + 1, as in the real config)
MINI_HIDDEN_SIZE = 256
MINI_NUM_HEADS = 8
MINI_NUM_LAYERS = 4
MINI_VOCAB_SIZE = 1024
MINI_MAX_POSITION = 4096

# layer_id -> compress_ratio. Index 4 is the MTP entry (never constructed here).
MINI_COMPRESS_RATIOS = [0, 4, 128, 0, 0]
LAYER_DENSE = 0
LAYER_CR4 = 1
LAYER_CR128 = 2

MINI_CONFIG_FIELDS: dict[str, Any] = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    # --- shrunk ---
    "hidden_size": MINI_HIDDEN_SIZE,
    "num_attention_heads": MINI_NUM_HEADS,
    "num_hidden_layers": MINI_NUM_LAYERS,
    "vocab_size": MINI_VOCAB_SIZE,
    "max_position_embeddings": MINI_MAX_POSITION,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    # --- structural: kept at real values ---
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "index_topk": 512,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "num_hash_layers": 3,
    "n_shared_experts": 1,
    "sliding_window": 128,
    "compress_ratios": MINI_COMPRESS_RATIOS,
    "expert_dtype": "fp4",
    "hc_mult": 4,
    "hc_eps": 1e-06,
    "hc_sinkhorn_iters": 20,
    "num_key_value_heads": 1,
    "num_nextn_predict_layers": 1,
    "norm_topk_prob": True,
    "rms_norm_eps": 1e-06,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "swiglu_limit": 10.0,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
    "attention_bias": False,
    "rope_theta": 10000,
    "compress_rope_theta": 160000,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    },
}

# transformers >= 4.56 exposes rope config as ``rope_parameters``; that is the
# attribute ``build_deepseek_v4_rope`` reads.
MINI_ROPE_PARAMETERS: dict[str, Any] = {
    "rope_type": "yarn",
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 16,
    "original_max_position_embeddings": 1024,
}

# TP used by these tests. TP1 keeps the harness single-process; the padded head
# count is TP-independent for any n_local_heads <= 64.
MINI_TP_SIZE = 1
MINI_N_LOCAL_HEADS = MINI_NUM_HEADS // MINI_TP_SIZE  # 8
EXPECTED_PADDED_HEADS = 64

# --------------------------------------------------------------------------- #
# expected parameter tables
# --------------------------------------------------------------------------- #
# H = 256, n_heads = 8, head_dim = 512, q_lora = o_lora = 1024, o_groups = 8,
# block quant = [128, 128] so a scale dim is ceil(size / 128).
_EXPECTED_BASE: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    # MergedColumnParallelLinear(256, [1024, 512], disable_tp=True)
    "fused_wqa_wkv.weight": ((1536, 256), FP8),
    "fused_wqa_wkv.weight_scale_inv": ((12, 2), E8M0),
    # RMSNorm(q_lora_rank)
    "q_norm.weight": ((1024,), BF16),
    # ColumnParallelLinear(1024, n_heads * head_dim = 4096)
    "wq_b.weight": ((4096, 1024), FP8),
    "wq_b.weight_scale_inv": ((32, 8), E8M0),
    # RMSNorm(head_dim)
    "kv_norm.weight": ((512,), BF16),
    # ColumnParallelLinear(n_heads * head_dim // o_groups = 512,
    #                      o_groups * o_lora_rank = 8192), is_bmm=True
    "wo_a.weight": ((8192, 512), FP8),
    "wo_a.weight_scale_inv": ((64, 4), E8M0),
    # RowParallelLinear(o_groups * o_lora_rank = 8192, 256)
    "wo_b.weight": ((256, 8192), FP8),
    "wo_b.weight_scale_inv": ((2, 64), E8M0),
    # No head padding -- attn_sink carries exactly n_local_heads values.
    # full((n_local_heads,), -inf, float32).
    "attn_sink": ((MINI_N_LOCAL_HEADS,), FP32),
}

# compressor: coff = 1 + (cr == 4); head_dim = 512
_EXPECTED_COMPRESSOR_CR4: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "compressor.ape": ((4, 1024), FP32),  # (cr, coff * head_dim)
    "compressor.fused_wkv_wgate.weight": ((2048, 256), BF16),  # 2 * coff * 512
    "compressor.norm.weight": ((512,), BF16),
}
_EXPECTED_COMPRESSOR_CR128: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "compressor.ape": ((128, 512), FP32),
    "compressor.fused_wkv_wgate.weight": ((1024, 256), BF16),
    "compressor.norm.weight": ((512,), BF16),
}

# indexer: only on cr == 4 layers. index_n_heads = 64, index_head_dim = 128,
# its inner compressor runs at head_dim = 128 with coff = 2.
_EXPECTED_INDEXER: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "indexer.wq_b.weight": ((8192, 1024), FP8),
    "indexer.wq_b.weight_scale_inv": ((64, 8), E8M0),
    "indexer.weights_proj.weight": ((64, 256), BF16),
    "indexer.compressor.ape": ((4, 256), FP32),
    "indexer.compressor.fused_wkv_wgate.weight": ((512, 256), BF16),
    "indexer.compressor.norm.weight": ((128,), BF16),
}

EXPECTED_PARAMS: dict[int, dict[str, tuple[tuple[int, ...], torch.dtype]]] = {
    LAYER_DENSE: dict(_EXPECTED_BASE),
    LAYER_CR128: {**_EXPECTED_BASE, **_EXPECTED_COMPRESSOR_CR128},
    LAYER_CR4: {**_EXPECTED_BASE, **_EXPECTED_COMPRESSOR_CR4, **_EXPECTED_INDEXER},
}


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session", autouse=True)
def _tp1_distributed_env():
    """Single-process TP1 group: parallel linear layers need an initialised TP
    group even when they are built on ``meta``. gloo keeps it CPU-only.

    ``initialize_model_parallel`` reads ``get_current_vllm_config()`` (community
    ``vllm/distributed/parallel_state.py``, the strict variant that asserts a
    config context is active) to pick up ``data_parallel_size`` /
    ``enable_elastic_ep``, so that one call must run inside
    ``set_current_vllm_config``. ``tests/conftest.py::default_vllm_config`` does
    exactly this, but it is function-scoped and a session fixture cannot depend
    on it, so a throwaway context is opened here instead. It wraps only the
    ``initialize_model_parallel`` call: the TP/PP groups it creates are module
    globals in ``parallel_state`` and stay valid after the context exits, and a
    stale config must not leak into the tests (each test builds its own mini
    VllmConfig via ``build_mini_fp8_attention``).

    ``init_distributed_environment`` is deliberately left outside the context:
    it uses ``get_current_vllm_config_or_none()`` and handles ``None`` (only the
    multi-node / DP>1 rank-offset branch consults the config, neither of which
    applies at TP1), so it needs no config context.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )
    from vllm.utils.network_utils import get_open_port

    if not model_parallel_is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
            backend="gloo",
        )
        with set_current_vllm_config(VllmConfig()):
            initialize_model_parallel(
                tensor_model_parallel_size=MINI_TP_SIZE,
                pipeline_model_parallel_size=1,
                backend="gloo",
            )
    yield


def make_mini_hf_config():
    """Mini DeepSeek-V4-Flash hf_config (structural fields kept, sizes shrunk)."""
    from transformers import PretrainedConfig

    # Deep copy: ``DeepseekV4ForCausalLMConfig.verify_and_update_model_config``
    # (replayed by ``make_mini_model_config``) rewrites
    # ``quantization_config["quant_method"]`` in place, and ``PretrainedConfig``
    # stores the nested dict by reference. Without the copy that write would
    # land in the module-level ``MINI_CONFIG_FIELDS`` and leak across tests.
    config = PretrainedConfig(**copy.deepcopy(MINI_CONFIG_FIELDS))
    # ``rope_parameters`` is consumed (and mutated) by build_deepseek_v4_rope;
    # hand it a private copy so repeated builds stay independent.
    config.rope_parameters = dict(MINI_ROPE_PARAMETERS)
    return config


# --------------------------------------------------------------------------- #
# ModelConfig for the mini config
# --------------------------------------------------------------------------- #
# A real ``ModelConfig`` cannot be built here: ``ModelConfig.__post_init__``
# resolves a HF repo (``get_config(self.hf_config_path or self.model, ...)``,
# ``vllm/config/model.py``) and asks ``ModelRegistry`` to inspect the model
# class in a *subprocess*, neither of which is available for a synthetic config
# with no checkpoint directory. So the object is built field by field -- but
# completely, not "the handful of fields the layer under test happens to read":
#
#   phase 1  every *declared* field <- its declared default, so a property that
#            reads a plain field finds exactly what the engine would;
#   phase 2  the values the CLI/engine would have supplied for this mini model;
#   phase 3  every attribute ``__post_init__`` *derives*, replayed in the same
#            order, computed with the community helper where that helper is pure
#            and hard-coded (with the reason) where the real one does I/O.
#
# ``test_ilu_dsv4_fp8_model_names.py`` guards phase 3 against vLLM bumps by
# diffing this against the ``self.<attr> = ...`` assignments it parses out of
# ``ModelConfig``'s own source, so a newly derived attribute shows up as a named
# assertion failure instead of an ``AttributeError`` deep inside a layer.
MINI_MODEL_NAME = "mini-deepseek-v4-flash"

# The methods whose ``self.<attr> = ...`` assignments together make up a fully
# initialised ``ModelConfig``; ``__post_init__`` calls the three ``_verify_*``
# ones at its very end.
MODEL_CONFIG_INIT_METHODS: tuple[str, ...] = (
    "__post_init__",
    "_verify_quantization",
    "_verify_cuda_graph",
    "_verify_bnb_config",
)

# ``_pooler_config_sources`` is assigned only on the ``runner_type == "pooling"``
# branch. A generative config -- ours *and* the engine's for DS-V4 -- never
# takes that branch, so it must not be required of the fixture either.
POOLING_ONLY_INIT_ATTRS = frozenset({"_pooler_config_sources"})


def post_init_assigned_attrs(cls=None) -> frozenset[str]:
    """``ModelConfig`` attributes assigned during a real initialisation.

    Parsed out of community source rather than listed by hand: the point is to
    notice when a vLLM bump starts deriving a *new* attribute in
    ``__post_init__``. Only ``self.<name> = ...`` targets count; nested writes
    (``self.hf_text_config.sliding_window = ...``) are somebody else's object.
    """
    if cls is None:
        from vllm.config import ModelConfig

        cls = ModelConfig

    assigned: set[str] = set()
    for method_name in MODEL_CONFIG_INIT_METHODS:
        source = textwrap.dedent(inspect.getsource(getattr(cls, method_name)))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    assigned.add(target.attr)
    return frozenset(assigned - POOLING_ONLY_INIT_ATTRS)


def _declared_field_defaults(cls) -> dict[str, Any]:
    """Every declared field of a vLLM config dataclass -> its declared default.

    Mirrors ``vllm.config.utils.get_field``: vLLM configs are *pydantic*
    dataclasses, so a field written as ``max_model_len: int = Field(default=None,
    ge=-1)`` carries a ``FieldInfo`` in ``dataclasses.Field.default`` instead of
    the value. Fields with neither default nor factory (``hf_config`` /
    ``hf_text_config``, both ``field(init=False)``) are skipped -- the caller
    sets those explicitly.
    """
    from pydantic.fields import FieldInfo

    defaults: dict[str, Any] = {}
    for field_info in dataclasses.fields(cls):
        default = field_info.default
        default_factory = field_info.default_factory
        if isinstance(default, FieldInfo):
            # same unwrapping order as get_field(): factory wins over default
            if default.default_factory is not None:
                defaults[field_info.name] = default.default_factory()
            elif not default.is_required():
                defaults[field_info.name] = default.default
        elif default_factory is not dataclasses.MISSING:
            defaults[field_info.name] = default_factory()
        elif default is not dataclasses.MISSING:
            defaults[field_info.name] = default
    return defaults


def make_mini_model_config(hf_config=None, *, model_cls=None):
    """A fully populated ``ModelConfig`` for the mini DS-V4-Flash config.

    ``model_cls`` is the class ``_ModelInfo`` is derived from; it defaults to
    ``IluDeepseekV4Fp8ForCausalLM``, which is what the engine resolves the
    ``DeepseekV4ForCausalLM`` architecture to on this platform.
    """
    from vllm.config import ModelConfig
    from vllm.model_executor.models.registry import _ModelInfo
    from vllm.transformers_utils.config import get_hf_text_config

    if hf_config is None:
        hf_config = make_mini_hf_config()
    if model_cls is None:
        from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_model import (
            IluDeepseekV4Fp8ForCausalLM,
        )

        model_cls = IluDeepseekV4Fp8ForCausalLM

    model_config = ModelConfig.__new__(ModelConfig)

    def put(name: str, value: Any) -> None:
        # object.__setattr__ to stay out of pydantic's assignment validation --
        # ``dtype`` is declared ``ModelDType | torch.dtype`` but holds a real
        # ``torch.dtype`` after __post_init__, and ``runner_type`` etc. are not
        # declared fields at all.
        object.__setattr__(model_config, name, value)

    # phase 1: declared fields <- declared defaults.
    for name, value in _declared_field_defaults(ModelConfig).items():
        put(name, value)

    # phase 2: what the engine would have been started with.
    put("model", MINI_MODEL_NAME)
    put("max_model_len", MINI_MAX_POSITION)
    put("enforce_eager", True)

    # phase 3: what __post_init__ derives.
    # served_model_name / tokenizer / tokenizer_revision: get_served_model_name
    # falls back to ``model``, and an unset tokenizer/revision mirrors the model.
    put("served_model_name", MINI_MODEL_NAME)
    put("tokenizer", MINI_MODEL_NAME)
    put("tokenizer_revision", None)
    # tokenizer_mode: __post_init__ resolves "auto" to "deepseek_v4" for the
    # DeepseekV4ForCausalLM architecture.
    put("tokenizer_mode", "deepseek_v4")
    put("hf_config", hf_config)
    put("hf_text_config", get_hf_text_config(hf_config))
    # model_arch_config: pure function of the two configs above (the convertor
    # only reads attributes), so the community dispatcher is called for real.
    put("model_arch_config", model_config.get_model_arch_config())
    put(
        "attention_chunk_size",
        getattr(model_config.hf_text_config, "attention_chunk_size", None),
    )
    # encoder_config / hf_image_processor_config: the real calls
    # (get_sentence_transformer_tokenizer_config / get_hf_image_processor_config)
    # go to the HF hub or the model directory. For a text-only, non
    # sentence-transformers model they yield exactly these two values.
    put("encoder_config", None)
    put("hf_image_processor_config", {})
    # runner_type / convert_type: ``--runner auto`` + ``--convert auto`` on a
    # text-generation architecture resolve to these. The real resolution goes
    # through get_pooling_config (hub) and ModelRegistry.inspect_model_cls
    # (subprocess), neither of which is usable for a synthetic model id.
    put("runner_type", "generate")
    put("convert_type", "none")
    # _model_info: pure ``issubclass``-style introspection of the model class,
    # so it is computed in-process here instead of via the registry subprocess.
    put("_model_info", _ModelInfo.from_model_cls(model_cls))
    put("_architecture", model_config.architectures[0])
    # dtype: _get_and_verify_dtype on an explicit bf16 request.
    put("dtype", BF16)
    # max_model_len: the derived cap (max_position_embeddings * yarn factor) is
    # far above the requested value, so get_and_verify_max_len returns it
    # unchanged. original_max_model_len is the pre-verification request.
    put("original_max_model_len", MINI_MAX_POSITION)
    put("max_model_len", MINI_MAX_POSITION)
    # quantization: _verify_quantization probes every registered method for an
    # override; DeepseekV4FP8Config.override_quantization_method claims a
    # ``quant_method == "fp8"`` config whose model_type is ``deepseek_v4`` and
    # reports get_name() == "deepseek_v4_fp8". Hard-coded because the probe
    # imports every quantization backend.
    put("quantization", "deepseek_v4_fp8")
    put("config_updated", False)
    # ...and the model-specific config hook, which is what rewrites the
    # checkpoint's ``quant_method`` from "fp8" to "deepseek_v4_fp8" in
    # hf_config. Pure (it only touches the config objects), so it runs for real.
    model_config._try_verify_and_update_model_config()

    return model_config


def make_mini_vllm_config():
    """A VllmConfig carrying the mini hf_config plus the DS-V4 FP8 quant config.

    Uses the **plugin** config ``IluDeepseekV4FP8Config`` rather than community
    ``DeepseekV4FP8Config``, because that is what a real Iluvatar run resolves to
    (it is registered under the same name ``deepseek_v4_fp8`` and shadows the
    community class). Only the routed-MXFP4 branch of ``get_quant_method``
    differs -- it returns ``IluMxfp4MoEMethod`` -- so every non-MoE layer built
    from this config, attention included, is unaffected.

    Note: with the community config, constructing a DS-V4 MoE layer
    off ROCm dies inside ``Mxfp4MoEMethod.__init__`` ->
    ``select_deepseek_v4_mxfp4_moe_backend``. That used to be worked around by
    registering patch P1 around every construction site; the plugin config makes
    the workaround unnecessary.
    """
    from vllm.config import VllmConfig

    from vllm_iluvatar.model_executor.layers.quantization.ilu_deepseek_v4_fp8 import (
        IluDeepseekV4FP8Config,
    )

    vllm_config = VllmConfig()
    vllm_config.model_config = make_mini_model_config()
    # fp8_ds_mla is a hard precondition of the DS-V4 attention layer
    # with cache_dtype == "auto" it asserts out.
    vllm_config.cache_config.cache_dtype = "fp8_ds_mla"
    vllm_config.cache_config.block_size = 64
    vllm_config.quant_config = IluDeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    return vllm_config


@contextlib.contextmanager
def build_mini_fp8_attention(layer_id: int, device: str = "meta"):
    """Build one ``IluDeepseekV4Fp8Attention`` on ``device`` under the mini config.

    Yields ``(layer, vllm_config)``. A fresh VllmConfig per call keeps
    ``static_forward_context`` prefixes from colliding across builds.
    """
    from vllm.config import set_current_vllm_config
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_attention import (
        IluDeepseekV4Fp8Attention,
    )

    vllm_config = make_mini_vllm_config()
    prefix = f"model.layers.{layer_id}.attn"
    # bf16 default dtype: the unquantised sub-modules (RMSNorm, compressor
    # fused_wkv_wgate, indexer weights_proj) take their dtype from it.
    # The vllm_config context stays active for the caller's body so follow-up
    # calls (e.g. get_kv_cache_spec) still see it; only the construction itself
    # runs under the target device.
    with set_default_torch_dtype(BF16), set_current_vllm_config(vllm_config):
        with torch.device(device):
            layer = IluDeepseekV4Fp8Attention(vllm_config, prefix)
        yield layer, vllm_config


def _named_param_table(layer) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    return {
        name: (tuple(param.shape), param.dtype)
        for name, param in layer.named_parameters()
    }


def _assert_todo_raises(layer, method_name: str, expected_tag: str) -> None:
    """Call ``method_name`` with placeholder args; it must raise its TODO."""
    method = getattr(layer, method_name)
    positional = [
        None
        for param in inspect.signature(method).parameters.values()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    with pytest.raises(NotImplementedError) as excinfo:
        method(*positional)
    message = str(excinfo.value)
    assert expected_tag in message, (
        f"{method_name} must raise NotImplementedError carrying {expected_tag}, "
        f"got: {message!r}"
    )
    assert TODO_RE.search(message), (
        f"{method_name} TODO message must match {TODO_RE.pattern}, got: {message!r}"
    )


# --------------------------------------------------------------------------- #
# 1 + 2. parameter names, shapes, dtypes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "layer_id",
    [LAYER_DENSE, LAYER_CR128, LAYER_CR4],
    ids=["dense", "cr128", "cr4"],
)
def test_attention_param_names_match_design_table(layer_id: int):
    expected = EXPECTED_PARAMS[layer_id]

    with build_mini_fp8_attention(layer_id) as (layer, _):
        actual = _named_param_table(layer)

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    assert not missing, f"layer {layer_id}: parameters missing: {missing}"
    assert not unexpected, (
        f"layer {layer_id}: unexpected parameters: {unexpected}"
    )


@pytest.mark.parametrize(
    "layer_id",
    [LAYER_DENSE, LAYER_CR128, LAYER_CR4],
    ids=["dense", "cr128", "cr4"],
)
def test_attention_param_shapes_and_dtypes_match_design_table(layer_id: int):
    expected = EXPECTED_PARAMS[layer_id]

    with build_mini_fp8_attention(layer_id) as (layer, _):
        actual = _named_param_table(layer)

    mismatches = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in sorted(set(expected) & set(actual))
        if actual[name] != expected[name]
    }
    assert not mismatches, f"layer {layer_id}: shape/dtype mismatches: {mismatches}"


def test_dense_layer_has_no_compressor_and_no_indexer():
    with build_mini_fp8_attention(LAYER_DENSE) as (layer, _):
        names = set(_named_param_table(layer))
        compressor_or_indexer = sorted(
            name
            for name in names
            if name.startswith("compressor.") or name.startswith("indexer.")
        )
        assert layer.compress_ratio == 1
        assert layer.compressor is None
        assert layer.indexer is None
    assert not compressor_or_indexer, (
        f"a dense layer must own no compressor/indexer params: {compressor_or_indexer}"
    )


def test_cr128_layer_has_compressor_without_indexer():
    with build_mini_fp8_attention(LAYER_CR128) as (layer, _):
        names = set(_named_param_table(layer))
        assert layer.compress_ratio == 128
        assert layer.compressor is not None
        assert layer.indexer is None
    assert any(name.startswith("compressor.") for name in names)
    assert not [name for name in names if name.startswith("indexer.")]


def test_cr4_layer_has_both_compressor_and_indexer():
    with build_mini_fp8_attention(LAYER_CR4) as (layer, _):
        names = set(_named_param_table(layer))
        assert layer.compress_ratio == 4
        assert layer.compressor is not None
        assert layer.indexer is not None
    assert any(name.startswith("compressor.") for name in names)
    assert any(name.startswith("indexer.") for name in names)


# --------------------------------------------------------------------------- #
# 3. attn_sink: padded shape + -inf init
# --------------------------------------------------------------------------- #
def test_get_padded_num_q_heads_pads_to_64():
    from vllm_iluvatar.model_executor.models.deepseek_v4.fp8_attention import (
        IluDeepseekV4Fp8Attention,
    )

    padded = IluDeepseekV4Fp8Attention.get_padded_num_q_heads(MINI_N_LOCAL_HEADS)
    assert padded == EXPECTED_PADDED_HEADS
    # the rule itself: 64 if n <= 64 else 128
    assert IluDeepseekV4Fp8Attention.get_padded_num_q_heads(64) == 64
    assert IluDeepseekV4Fp8Attention.get_padded_num_q_heads(65) == 128


def test_attn_sink_shape_is_local_head_count():
    """No head padding -- attn_sink is n_local_heads."""
    with build_mini_fp8_attention(LAYER_CR4) as (layer, _):
        assert layer.n_local_heads == MINI_N_LOCAL_HEADS
        assert tuple(layer.attn_sink.shape) == (MINI_N_LOCAL_HEADS,)
        assert layer.attn_sink.dtype is FP32


def test_attn_sink_is_initialised_to_negative_inf():
    """Construction-time init only. Weight loading fills all ``n_local_heads``
    slots (no padding); weight loading fills all ``n_local_heads`` slots."""
    # Values need real storage, so build on CPU. The dense flavour is used
    # because it pulls in neither compressor nor indexer.
    with build_mini_fp8_attention(LAYER_DENSE, device="cpu") as (layer, _):
        attn_sink = layer.attn_sink.detach()

    assert attn_sink.shape == (MINI_N_LOCAL_HEADS,)
    assert attn_sink.dtype is FP32
    assert torch.isneginf(attn_sink).all(), (
        f"attn_sink must start as all -inf, got {attn_sink}"
    )


# --------------------------------------------------------------------------- #
# 4. forward entries are TODO-gated
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "method_name, expected_tag",
    [
        ("forward", "TODO(ilu_fp8_A1)"),
        ("forward_mqa", "TODO(ilu_fp8_A2)"),
        ("_o_proj", "TODO(ilu_fp8_A3)"),
        ("_fused_qnorm_rope_kv_insert", "TODO(ilu_fp8_A4)"),
    ],
)
def test_forward_entries_raise_numbered_todo(method_name: str, expected_tag: str):
    with build_mini_fp8_attention(LAYER_CR4) as (layer, _):
        _assert_todo_raises(layer, method_name, expected_tag)

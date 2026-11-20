# SPDX-License-Identifier: Apache-2.0
"""GLM5Next KDA OOT kernels with the upstream cache and projection contracts."""

from __future__ import annotations

import ixformer.inference.functions as ixfops
import torch
import vllm.envs as envs
from ixformer.inference.backend_registry import registry as ix_registry
from torch import nn
from torch.nn.functional import linear
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.glm5next.common.kda import (
    _Glm5NextMergedColumnParallelLinear,
)
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig

from vllm_iluvatar.ilu_util.arch import is_ivcore11
from vllm_iluvatar.ilu_util.streams import get_prefill_aux_stream
from vllm_iluvatar.model_executor.layers.mamba.ops.ilu_causal_conv1d import (
    causal_conv1d_fn,
)
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.schemes.ilu_compressed_tensors_w8a8_int8 import (  # noqa: E501
    IluW8A8OnlineLinearMethod,
)
from vllm_iluvatar.v1.attention.backends.ilu_gdn_attn import IluGDNAttentionMetadata

logger = init_logger(f"vllm.{__name__}")


def _project_gates_async(x, weight, gate_sizes, f_projection, g_projection, shape):
    caller = torch.cuda.current_stream(x.device)
    stream = get_prefill_aux_stream(x.device)
    stream.wait_stream(caller)
    x.record_stream(stream)
    with torch.cuda.stream(stream):
        beta, f_a, g_a = linear(x, weight).split(gate_sizes, dim=-1)
        g1 = f_projection(f_a)[0].view(shape)
        g2 = g_projection(g_a)[0].view(shape)
        ready = stream.record_event()
    for output in (beta, g1, g2):
        output.record_stream(caller)
    return beta, g1, g2, ready


class IluGlm5NextLinearAttention(GatedDeltaNetAttention):
    head_dim: int
    num_heads: int
    conv_size: int

    def get_attn_backend(self):
        if self.cache_config.use_kda_recoverssm:
            from vllm_iluvatar.v1.attention.backends.ilu_glm5next_kda import (
                IluGlm5NextKDAAttentionBackend,
            )

            return IluGlm5NextKDAAttentionBackend
        return super().get_attn_backend()

    def bind_kv_cache(self, kv_cache):
        ref = getattr(self, "_recoverssm_resources_ref", None)
        resources = ref() if ref is not None else None
        if resources is not None:
            resources.before_layer_binding()
        super().bind_kv_cache(kv_cache)
        if resources is not None:
            resources.after_layer_binding()

    @torch.no_grad()
    def _prepare_recoverssm_parameters(self):
        if not self.use_recoverssm:
            return
        ref = self._recoverssm_resources_ref
        resources = ref() if ref is not None else None
        if (
            resources is not None
            and resources.captured_generation is not None
            and (
                self.A_log_view.data_ptr() != self.A_log.data_ptr()
                or self.dt_bias_view.data_ptr() != self.dt_bias.data_ptr()
            )
        ):
            raise RuntimeError(
                "Recapture GLM graphs after KDA parameter storage changes"
            )
        self.A_log_view = self.A_log.reshape(self.local_num_heads)
        self.dt_bias_view = self.dt_bias.reshape(self.local_num_heads, self.head_dim)
        for i, module in enumerate((self.q_conv1d, self.k_conv1d, self.v_conv1d)):
            self._merged_conv_weight[
                i * self.local_projection_size : (i + 1) * self.local_projection_size
            ].copy_(module.weight.reshape(self.local_projection_size, 4))
        self._recoverssm_weights_ready = True

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Prepare paired dense weights after the per-linear loading hooks."""
        if getattr(self, "use_recoverssm", False):
            self._prepare_recoverssm_parameters()
        previous = self._fg_b_weight
        self._fg_b_weight = None
        if ix_registry.resolve("bmm") != "ixmm":
            return
        f, g = self.f_b_proj, self.g_b_proj
        if not all(
            isinstance(layer.quant_method, UnquantizedLinearMethod)
            and layer.bias is None
            for layer in (f, g)
        ):
            return
        if (
            f.weight.dtype not in (torch.float16, torch.bfloat16)
            or f.weight.dtype != act_dtype
            or g.weight.dtype != f.weight.dtype
            or f.weight.shape != g.weight.shape
            or self.head_dim % 32 != 0
            or self.local_projection_size % 2 != 0
            or (self.local_num_heads * f.weight.element_size()) % 64 != 0
        ):
            return
        # The B/F/G tail starts with beta. Its width determines both the
        # f/g starting offset and row alignment required by ixmm.
        if (
            previous is not None
            and previous.shape == (2, *f.weight.shape)
            and previous.dtype == f.weight.dtype
            and previous.device == f.weight.device
        ):
            paired = previous
            for index, weight in enumerate((f.weight, g.weight)):
                if weight.data_ptr() != paired[index].data_ptr():
                    paired[index].copy_(weight.detach())
        else:
            paired = torch.stack((f.weight.detach(), g.weight.detach()))
        # Preserve parameter names, TP loading attributes and reload behavior.
        # The original Parameters become contiguous views of the pair, so there
        # is no additional persistent copy of the projection weights.
        f.weight.data = paired[0]
        g.weight.data = paired[1]
        self._fg_b_weight = paired
        logger.info_once("GLM5Next KDA paired gate projections enabled")

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: Glm5NextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        saved_quant_config = vllm_config.quant_config
        try:
            vllm_config.quant_config = None
            super().__init__(config, vllm_config, prefix)
        finally:
            vllm_config.quant_config = saved_quant_config

        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.conv_size = config.linear_conv_kernel_dim
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(projection_size, self.tp_size)

        _, recurrent_state_shape = self.get_state_shape()
        _, recurrent_state_dtype = self.get_state_dtype()
        scatter_states.register_warmup(
            state_shape=recurrent_state_shape,
            state_dtype=recurrent_state_dtype,
            indices_dtype=torch.int32,
            max_num_tokens=vllm_config.scheduler_config.max_num_seqs,
        )

        self.in_proj_qkvbfg_a = _Glm5NextMergedColumnParallelLinear(
            self.hidden_size,
            [
                projection_size,
                projection_size,
                projection_size,
                self.num_heads,
                self.head_dim,
                self.head_dim,
            ],
            replicated_shard_ids=(4, 5),
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvbfg_a",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )
        self.dt_bias_view = self.dt_bias.view(self.local_num_heads, self.head_dim)
        assert self.dt_bias_view.data_ptr() == self.dt_bias.data_ptr()

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)
        self._merged_conv_weight: torch.Tensor | None = None

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})
        self.A_log_view = self.A_log.reshape(self.local_num_heads)
        assert self.A_log_view.data_ptr() == self.A_log.data_ptr()

        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.register_buffer("_fg_b_weight", None, persistent=False)
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self._split_qkvbfg_output = envs.VLLM_ILU_GLM5NEXT_KDA_PROJ_W8A8
        if self._split_qkvbfg_output:
            self.in_proj_qkvbfg_a.quant_method = IluW8A8OnlineLinearMethod(
                self.in_proj_qkvbfg_a.quant_method,
                quantized_rows=3 * self.local_projection_size,
                cat_output=False,
            )
            self.o_proj.quant_method = IluW8A8OnlineLinearMethod(
                self.o_proj.quant_method
            )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        def _a_log_weight_loader(param, loaded_weight):
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view([1, 1, -1, 1])
            return sharded_weight_loader(2)(param, loaded_weight)

        self.A_log.weight_loader = _a_log_weight_loader

        self.kda_safe_gate = True
        self.kda_lower_bound = config.linear_lower_bound
        self._conv_state_dim_first = is_conv_state_dim_first()

        self.use_recoverssm = self.cache_config.use_kda_recoverssm
        self._recoverssm_resources_ref = None
        if self.use_recoverssm:
            rows = vllm_config.scheduler_config.max_num_seqs
            t = self.num_spec + 1
            self.register_buffer(
                "_recoverssm_correction",
                torch.empty(
                    rows,
                    self.local_num_heads,
                    t,
                    128,
                    dtype=torch.float32,
                    device=self.A_log.device,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_recoverssm_key_gate",
                torch.empty(
                    rows,
                    self.local_num_heads,
                    t,
                    256,
                    dtype=torch.bfloat16,
                    device=self.A_log.device,
                ),
                persistent=False,
            )
            del self._merged_conv_weight
            self.register_buffer(
                "_merged_conv_weight",
                torch.empty(
                    3 * self.local_projection_size,
                    4,
                    dtype=torch.float32,
                    device=self.A_log.device,
                ),
                persistent=False,
            )
            self._recoverssm_weights_ready = False

    def _project_qkv_and_gates(self, hidden_states: torch.Tensor):
        projection = self.in_proj_qkvbfg_a
        method = projection.quant_method
        if (
            hidden_states.ndim == 2
            and isinstance(method, IluW8A8OnlineLinearMethod)
            and method.quantized_rows == 3 * self.local_projection_size
            and projection.bias is None
            and not projection.gather_output
        ):
            # This merged column-parallel layer has no bias or output gather.
            # Keep QKV and the BF16 B/F/G tail in their existing separate buffers.
            qkv, tail = method.apply_parts(self.in_proj_qkvbfg_a, hidden_states)
            beta_raw, fg = tail.split([self.local_num_heads, 2 * self.head_dim], dim=-1)
        else:
            projected = self.in_proj_qkvbfg_a(hidden_states)[0]
            if isinstance(projected, tuple):
                qkv, tail = projected
                beta_raw, fg = tail.split(
                    [self.local_num_heads, 2 * self.head_dim], dim=-1
                )
            else:
                qkv, beta_raw, fg = projected.split(
                    [
                        3 * self.local_projection_size,
                        self.local_num_heads,
                        2 * self.head_dim,
                    ],
                    dim=-1,
                )
        shape = (-1, self.local_num_heads, self.head_dim)
        if self._fg_b_weight is not None:
            # f_a/g_a are adjacent column slices. view/transpose expose a
            # non-overlapping [2, tokens, head_dim] batch without copying.
            paired_input = fg.view(-1, 2, self.head_dim).transpose(0, 1)
            gates = ixfops.bmm(paired_input, self._fg_b_weight, format="TN")
            g1, g2 = gates[0].view(shape), gates[1].view(shape)
        else:
            f_a, g_a = fg.split(self.head_dim, dim=-1)
            g1 = self.f_b_proj(f_a)[0].view(shape)
            g2 = self.g_b_proj(g_a)[0].view(shape)
        return qkv, beta_raw, g1, g2

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        shape = (-1, self.local_num_heads, self.head_dim)
        ready = None
        projection = self.in_proj_qkvbfg_a
        method = projection.quant_method
        gate_sizes = [self.local_num_heads, self.head_dim, self.head_dim]
        if (
            self.tp_size == 1
            and hidden_states.ndim == 2
            and hidden_states.shape[0] >= 512
            and is_ivcore11()
            and isinstance(method, IluW8A8OnlineLinearMethod)
            and method.quantized_rows == 3 * self.local_projection_size
            and projection.bias is None
            and not projection.gather_output
            and not torch.cuda.is_current_stream_capturing()
            and not torch.compiler.is_compiling()
        ):
            attn = get_forward_context().attn_metadata
            metadata = attn.get(self.prefix) if isinstance(attn, dict) else None
            if (
                isinstance(metadata, IluGDNAttentionMetadata)
                and metadata.num_prefills > 0
                and metadata.num_decodes == metadata.num_spec_decodes == 0
            ):
                logger.info_once(
                    "ILU KDA Prefill gate stream active on %s", hidden_states.device
                )
                beta_raw, g1, g2, ready = _project_gates_async(
                    hidden_states,
                    projection.bf16_tail_weight,
                    gate_sizes,
                    self.f_b_proj,
                    self.g_b_proj,
                    shape,
                )
                qkv = method.scheme.apply_weights(projection, hidden_states, bias=None)

        if ready is None:
            qkv, beta_raw, g1, g2 = self._project_qkv_and_gates(hidden_states)

        core_attn_out = torch.empty_like(g1)

        self._forward(
            qkv_proj_states=qkv,
            g1=g1,
            beta=beta_raw,
            core_attn_out=core_attn_out,
            gate_ready=ready,
        )
        if ready is not None:
            ready.wait(torch.cuda.current_stream(core_attn_out.device))
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), -1)
        return self.o_proj(core_attn_out)[0]

    @eager_break_during_capture
    def _forward(
        self,
        qkv_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
        gate_ready: torch.cuda.Event | None = None,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            core_attn_out.zero_()
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
        if attn_metadata_narrowed is None:
            core_attn_out.zero_()
            return

        assert isinstance(attn_metadata_narrowed, IluGDNAttentionMetadata)
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_query_start_loc_cpu = (
            attn_metadata_narrowed.non_spec_query_start_loc_cpu
        )
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        spec_sequence_masks = attn_metadata_narrowed.spec_sequence_masks
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        num_spec_decodes = attn_metadata_narrowed.num_spec_decodes
        use_spec = num_spec_decodes > 0 and (
            self.use_recoverssm or spec_sequence_masks is not None
        )

        lower_bound = self.kda_lower_bound
        constant_caches = self.kv_cache

        # The KDA kernels only write the rows their token indices cover, so the
        # graph-padding tail is this function's to clear. ``core_attn_out`` is
        # uninitialised on entry; ``o_norm`` reads all of it.
        if num_actual_tokens < core_attn_out.shape[0]:
            core_attn_out[num_actual_tokens:].zero_()

        qkv_proj_states = qkv_proj_states[:num_actual_tokens]
        g1 = g1[:num_actual_tokens]
        beta = beta[:num_actual_tokens]

        (conv_state, recurrent_state) = constant_caches
        if not self._conv_state_dim_first:
            conv_state = conv_state.transpose(-1, -2)

        if self._merged_conv_weight is None:

            def _w(m):
                return m.weight.view(m.weight.size(0), m.weight.size(2))

            self._merged_conv_weight = torch.cat(
                [_w(self.q_conv1d), _w(self.k_conv1d), _w(self.v_conv1d)],
                dim=0,
            ).contiguous()

        conv_weights = self._merged_conv_weight
        conv_bias = self.q_conv1d.bias

        if use_spec:
            q, k, v = qkv_proj_states.split(self.local_projection_size, dim=-1)
            if self.use_recoverssm:
                ixfops.kda_recoverssm_verify(
                    q,
                    k,
                    v,
                    g1,
                    beta,
                    self.A_log_view,
                    self.dt_bias_view,
                    recurrent_state,
                    self._recoverssm_correction,
                    self._recoverssm_key_gate,
                    spec_query_start_loc[: num_spec_decodes + 1],
                    spec_state_indices_tensor[:num_spec_decodes, 0],
                    spec_query_len=self.num_spec + 1,
                    lower_bound=lower_bound,
                    out=core_attn_out,
                    record_rows=attn_metadata_narrowed.recoverssm_record_rows,
                    token_index=spec_token_indx,
                    conv_weight=conv_weights,
                    conv_state_pool=conv_state,
                    pad_slot_id=0,
                )
            else:
                ixfops.fused_recurrent_kda(
                    q,
                    k,
                    v,
                    g1,
                    beta,
                    self.A_log_view,
                    self.dt_bias_view,
                    core_attn_out,
                    recurrent_state,
                    spec_query_start_loc[: num_spec_decodes + 1],
                    spec_state_indices_tensor,
                    self.head_dim**-0.5,
                    lower_bound,
                    None,
                    num_accepted_tokens,
                    spec_token_indx,
                    conv_weight=conv_weights,
                    conv_state_pool=conv_state,
                    pad_slot_id=0,
                )

        if attn_metadata_narrowed.num_prefills > 0:
            qkv_ns = causal_conv1d_fn(
                qkv_proj_states.transpose(0, 1),
                conv_weights,
                conv_bias,
                activation="silu",
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
                input_index=non_spec_token_indx,
            ).transpose(0, 1)
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)
            if gate_ready is not None:
                gate_ready.wait(torch.cuda.current_stream(q_ns.device))
            ixfops.chunk_kda_with_fused_gate(
                q_ns,
                k_ns,
                v_ns,
                g1,
                beta,
                self.A_log_view,
                self.dt_bias_view,
                core_attn_out,
                recurrent_state,
                non_spec_query_start_loc,
                non_spec_query_start_loc_cpu,
                non_spec_state_indices_tensor,
                has_initial_state,
                self.head_dim**-0.5,
                lower_bound,
                None,
                non_spec_token_indx,
            )
        elif attn_metadata_narrowed.num_decodes > 0:
            q, k, v = qkv_proj_states.split(self.local_projection_size, dim=-1)
            ixfops.fused_recurrent_kda(
                q,
                k,
                v,
                g1,
                beta,
                self.A_log_view,
                self.dt_bias_view,
                core_attn_out,
                recurrent_state,
                non_spec_query_start_loc,
                non_spec_state_indices_tensor,
                self.head_dim**-0.5,
                lower_bound,
                None,
                None,
                None,
                conv_weight=conv_weights,
                conv_state_pool=conv_state,
                pad_slot_id=0,
            )

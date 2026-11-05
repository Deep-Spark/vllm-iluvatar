# Source: ixaite_compressed_tensors_w8a8_int8.py at 47fb804; keep loading, padding, bind and apply together.
# SPDX-License-Identifier: Apache-2.0
"""ixaite W8A8 INT8 compressed-tensors linear scheme.

The independently selected ``ixaite`` backend uses it for supported channel-wise schemes, including asymmetric/AZP
activation quantization. Decode calls ``ixaite.ops.w8a8.apply`` with a cached
launcher; it does not go through ``torch.ops`` or fall back to ixformer.
"""

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationStrategy
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (  # noqa: E501
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

from vllm_iluvatar.backends.ixaite import ops
from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op
from vllm_iluvatar.model_executor.layers.quantization.compressed_tensors.ilu_utils import (  # noqa: E501
    align_up,
    pad_tensor,
    replace_with_padded,
    unpad_tensor,
)

logger = init_logger(__name__)

_current_stream = torch.cuda.current_stream

_ALIGN = 64
_FORMAT_TO_MAJOR = {"NN": "tt", "TN": "tn"}
_OUT_KIND = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
}


def _warmup_row_bound() -> int | None:
    """Largest M the scheduler can hand the GEMM, or None in unit tests."""
    try:
        from vllm.config import get_current_vllm_config

        sched = get_current_vllm_config().scheduler_config
    except Exception:
        return None
    bound = max(
        getattr(sched, "max_num_batched_tokens", 0) or 0,
        getattr(sched, "max_num_seqs", 0) or 0,
    )
    return bound or None


class IxaiteCompressedTensorsW8A8Int8(CompressedTensorsScheme):
    """Channel-wise W8A8 linear scheme on ixaite FlyDSL kernels."""

    def __init__(
        self,
        strategy: str,
        is_static_input_scheme: bool,
        input_symmetric: bool,
    ):
        # Standard checkpoint scales determine the layout; vendor options do not.
        self.strategy = strategy
        if self.strategy != QuantizationStrategy.CHANNEL:
            raise ValueError(
                "IxaiteCompressedTensorsW8A8Int8 only supports channel-wise "
                "quantization"
            )
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric
        self.params_dtype: torch.dtype | None = None
        self._apply = None
        self._launch = None
        self._m = -1
        self._n = 0
        self._k = 0
        self._pad_n = 0
        self._pad_k = 0
        self._major = "tt"
        self._out_kind = "bf16"
        self._apply_bias = False

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        layer.logical_widths = output_partition_sizes
        self.params_dtype = params_dtype

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", weight_scale)

        input_scale = None
        input_zero_point = None
        if self.is_static_input_scheme:
            input_scale = BasevLLMParameter(
                data=torch.empty(1, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            if not self.input_symmetric:
                input_zero_point = BasevLLMParameter(
                    data=torch.empty(1, dtype=torch.int8),
                    weight_loader=weight_loader,
                )
        layer.register_parameter("input_scale", input_scale)
        layer.register_parameter("input_zero_point", input_zero_point)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        apply_fn, reason = load_ixaite_python_op("w8a8", "apply")
        if apply_fn is None:
            raise RuntimeError(f"ixaite W8A8 apply is unavailable: {reason}")
        bind_fn, _ = load_ixaite_python_op("w8a8", "bind")
        warmup_fn, _ = load_ixaite_python_op("w8a8", "warmup")
        if bind_fn is None or warmup_fn is None:
            raise RuntimeError(
                "ixaite W8A8 bind/warmup is unavailable after apply resolved"
            )
        self._apply = apply_fn

        if self.params_dtype not in _OUT_KIND:
            raise ValueError(
                "IxaiteCompressedTensorsW8A8Int8 supports bf16/fp16 output, "
                f"got {self.params_dtype}"
            )
        self._out_kind = _OUT_KIND[self.params_dtype]

        weight = layer.weight
        n, k = weight.shape
        if n % _ALIGN == 0 and k % _ALIGN == 0:
            format_str = "NN"
            replace_parameter(
                layer,
                "weight",
                torch.nn.Parameter(weight.data.t().contiguous(), requires_grad=False),
            )
        else:
            format_str = "TN"

        self._n = n
        self._k = k
        self._pad_n = align_up(n, _ALIGN)
        self._pad_k = align_up(k, _ALIGN)
        self._format = format_str
        self._major = _FORMAT_TO_MAJOR[format_str]
        self._apply_bias = getattr(layer, "bias", None) is not None
        self._launch = None
        self._m = -1

        if format_str == "NN":
            target = (self._pad_k, self._pad_n)
        else:
            target = (self._pad_n, self._pad_k)
        replace_with_padded(layer, "weight", target, pad_value=0)

        if getattr(layer, "bias", None) is not None:
            replace_with_padded(layer, "bias", (self._pad_n,), pad_value=0)

        if self.strategy == QuantizationStrategy.CHANNEL:
            ws = layer.weight_scale
            if ws.shape[0] != self._pad_n:
                replace_with_padded(
                    layer,
                    "weight_scale",
                    (self._pad_n, ws.shape[1]),
                    pad_value=1,
                )
            ws = layer.weight_scale
            if ws.dim() == 2 and ws.shape[1] == 1:
                ws = ws.data.view(-1)
            if ws.dtype != torch.float32:
                ws = ws.to(torch.float32)
            replace_parameter(
                layer,
                "weight_scale",
                torch.nn.Parameter(
                    ws if ws.is_contiguous() else ws.contiguous(),
                    requires_grad=False,
                ),
            )

        if self.is_static_input_scheme:
            if self.input_symmetric:
                replace_parameter(
                    layer,
                    "input_scale",
                    torch.nn.Parameter(layer.input_scale.max(), requires_grad=False),
                )
            else:
                self._collapse_static_asymmetric_input(layer)

        if not self.input_symmetric:
            self._build_azp_adj(layer, format_str)

        bind_fn(
            n=self._pad_n,
            k=self._pad_k,
            out_kind=self._out_kind,
            apply_bias=self._apply_bias,
            major=self._major,
        )
        m_max = _warmup_row_bound()
        if m_max is None:
            logger.warning(
                "Ixaite W8A8 has no scheduler config; skipping warmup, "
                "kernels will compile on first use"
            )
            return
        try:
            built = warmup_fn(
                weight=layer.weight,
                m_max=m_max,
                out_dtype=self.params_dtype,
                apply_bias=self._apply_bias,
                format=format_str,
                apply_azp=not self.input_symmetric,
            )
        except Exception as exc:
            raise RuntimeError(
                f"ixaite W8A8 warmup failed for N={self._pad_n} K={self._pad_k}: {exc}"
            ) from exc
        logger.info_once(
            "Ixaite W8A8 warmup built %d M tiers (m_max:%d) per weight shape",
            len(built),
            m_max,
        )

    @staticmethod
    def _collapse_static_asymmetric_input(layer: torch.nn.Module) -> None:
        """Rebuild the per-tensor scale/zero point from the loaded per-channel pair.

        A checkpoint stores one (scale, zero point) per output partition; the
        kernel takes a single pair, so reconstruct the widest representable
        range across partitions and requantize it.
        """
        int8_traits = torch.iinfo(torch.int8)
        input_scale = layer.input_scale
        azps = layer.input_zero_point.to(dtype=torch.int32)
        range_max = (input_scale * (int8_traits.max - azps)).max()
        range_min = (input_scale * (int8_traits.min - azps)).min()

        scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)
        replace_parameter(
            layer, "input_scale", torch.nn.Parameter(scale, requires_grad=False)
        )
        azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)
        replace_parameter(
            layer, "input_zero_point", torch.nn.Parameter(azp, requires_grad=False)
        )

    def _build_azp_adj(self, layer: torch.nn.Module, format_str: str) -> None:
        """Precompute ``azp_adj[n] = sum_k w_q[n, k]`` over the padded weight.

        Reading the padded weight keeps the N pad entries at zero and leaves the
        K pad out of the sum, which is what makes zero-padded activations agree
        with the correction term.
        """
        weight = layer.weight
        # NN stores (pad_k, pad_n); TN stores (pad_n, pad_k).
        reduce_dim = 0 if format_str == "NN" else 1
        azp_adj = weight.to(torch.int32).sum(dim=reduce_dim)
        if self.is_static_input_scheme:
            # Per-tensor: folding the zero point in turns the correction into a
            # pure per-column term, so the kernel can take a row vector of ones.
            azp_adj = layer.input_zero_point * azp_adj
        layer.register_parameter(
            "azp_adj",
            torch.nn.Parameter(azp_adj.to(torch.float32), requires_grad=False),
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        w_q = layer.weight
        w_s = layer.weight_scale

        if isinstance(x, tuple):
            x_q, x_s, out_dtype = x
            x_zp = None
        else:
            out_dtype = x.dtype
            x_q, x_s, x_zp = ops.scaled_int8_quant(
                x.contiguous(),
                getattr(layer, "input_scale", None),
                getattr(layer, "input_zero_point", None),
                symmetric=self.input_symmetric,
            )

        if x_q.dim() not in (2, 3):
            raise ValueError(
                "IxaiteCompressedTensorsW8A8Int8 only supports 2D or 3D "
                f"inputs, got {x_q.dim()}D"
            )
        if x_q.shape[-1] != self._pad_k:
            if x_q.dim() == 2:
                x_q = pad_tensor(x_q, self._pad_k)
            else:
                x_q = torch.nn.functional.pad(
                    x_q, (0, self._pad_k - x_q.shape[-1]), value=0
                )

        out = torch.empty(
            (*x_q.shape[:-1], self._pad_n),
            dtype=out_dtype,
            device=x_q.device,
        )
        m = x_q.shape[0] if x_q.dim() == 2 else x_q.numel() // x_q.shape[-1]
        # A static scheme quantizes with one per-tensor scale, but the GEMM
        # epilogue reads scale_a per row; hand it one entry per row.
        if x_s.numel() == 1 and m != 1:
            x_s = x_s.reshape(1).expand(m).contiguous()
        apply_bias = bias is not None
        azp_adj = getattr(layer, "azp_adj", None)
        if azp_adj is None:
            azp = None
        elif self.is_static_input_scheme:
            # The zero point is already folded into azp_adj, so the per-row term
            # is a constant 1.
            azp = torch.ones(m, dtype=torch.float32, device=x_q.device)
        else:
            azp = x_zp
        launch = (
            self._launch if m == self._m and apply_bias == self._apply_bias else None
        )
        out_kind = self._out_kind
        if out_dtype is not self.params_dtype:
            try:
                out_kind = _OUT_KIND[out_dtype]
            except KeyError as exc:
                raise ValueError(
                    "IxaiteCompressedTensorsW8A8Int8 supports bf16/fp16 "
                    f"output, got {out_dtype}"
                ) from exc
            launch = None
        # Cached 2-D launch skips apply(). That wrapper still reshapes, looks
        # up the compile cache, and fills ctypes slots before the kernel can
        # start. B=1 prefill issues this GEMM ~160 times at M=32/128; the extra
        # Python is launch-bound against ixformer's one C++ ixf_ops.w8a8 hop.
        # The kernel is already packed; only pointers, M, and grid change.
        if torch.compiler.is_compiling():
            # vLLM piecewise compilation traces with fullgraph=True; the
            # FlyDSL compile/launch machinery below is untraceable, so call
            # the opaque custom-op form (same kernel via ixaite.ops.w8a8).
            torch.ops.ixaite.w8a8(
                x_q,
                w_q,
                x_s,
                w_s,
                out,
                bias,
                self._format,
                azp,
                azp_adj,
            )
        elif launch is not None and x_q.dim() == 2:
            sa = x_s if x_s.ndim == 1 else x_s.view(-1)
            sb = w_s if w_s.ndim == 1 else w_s.view(-1)
            if azp is None:
                launch(x_q, w_q, sa, sb, out, bias, _current_stream())
            else:
                za = azp if azp.ndim == 1 else azp.view(-1)
                launch(
                    x_q,
                    w_q,
                    sa,
                    sb,
                    out,
                    bias,
                    _current_stream(),
                    za.to(torch.float32),
                    azp_adj,
                )
            self._launch = launch
        else:
            self._launch = self._apply(
                x_q,
                w_q,
                x_s,
                w_s,
                out,
                n=self._pad_n,
                k=self._pad_k,
                major=self._major,
                out_kind=out_kind,
                bias=bias,
                launch=launch,
                azp=None if azp is None else azp.to(torch.float32),
                azp_adj=azp_adj,
            )
        self._m = m
        self._apply_bias = apply_bias

        if out.shape[-1] == self._n:
            result = out
        elif out.dim() == 2:
            result = unpad_tensor(out, self._n, contiguous=True)
        else:
            result = out[..., : self._n].contiguous()
        return result

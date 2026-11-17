# SPDX-License-Identifier: Apache-2.0
"""Runtime expert contracts, independent of checkpoint and vLLM Method classes."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import torch


@dataclass(frozen=True)
class MoEWeightSpec:
    format_id: str
    layout: str
    weight_dtype: torch.dtype
    scale_dtype: torch.dtype = torch.float32
    scale_group_size: int = -1
    # IxInfer v2 interprets INT4 codes as multiples of 18; IxMM does not.
    code_multiplier: int = 1
    alignment: int = 256

    def shapes(self, experts, n, k):
        if self.scale_group_size == 256:
            k = (k + 255) // 256 * 256
            return (experts, k // 2, n), (experts, k // 256, n)
        weight = (experts, n, k // 2) if self.layout == "TN" else (experts, k, n // 2)
        return weight, (experts, 1, n)


@dataclass(frozen=True)
class MoEWeights:
    """Non-owning parameter views: no clone, conversion, or checkpoint state."""

    spec: MoEWeightSpec
    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor

    @classmethod
    def from_layer(cls, layer, spec):
        return cls(
            spec,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
        )


@dataclass(frozen=True)
class MoEComputeConfig:
    hidden: int
    intermediate: int
    local_num_experts: int
    global_num_experts: int
    ep_rank: int = 0
    activation: str = "silu"
    limit: float | None = None
    output_scale: float = 1.0
    apply_router_weight_on_input: bool = False


@dataclass(frozen=True)
class MoEWorkspaceSpec:
    buffers: dict[str, tuple[tuple[int, ...], torch.dtype]]

    def allocate(self, device):
        return MoEWorkspace(self, device)


class MoEWorkspace:
    """Call-scoped lazy scratch. Graph captures own their allocator-pool storage.

    No layer/global cache retains buffers between calls or graph captures.
    Individual operators may also allocate their returned tensors internally.
    """

    def __init__(self, spec, device):
        self.spec = spec
        self.device = device
        self.buffers = {}

    def get(self, name, rows=None):
        shape, dtype = self.spec.buffers[name]
        if rows is not None and not 0 <= rows <= shape[0]:
            raise ValueError(f"{name}: {rows} exceeds workspace capacity {shape[0]}")
        if name not in self.buffers:
            self.buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)
        value = self.buffers[name]
        return value if rows is None else value[:rows]

    def release(self, *names):
        for name in names:
            self.buffers.pop(name, None)


class MoEComputeBackend(ABC):
    """One instance per layer; holds static execution data, never the layer."""

    # Each implementation declares capabilities once. Tuple order defines its
    # default layout; priority makes automatic choice independent of import order.
    name: ClassVar[str]
    supported_layouts: ClassVar[Mapping[str, tuple[str, ...]]]
    weight_specs: ClassVar[Mapping[str, MoEWeightSpec]]
    priority: ClassVar[int]
    activation_dtype: ClassVar[torch.dtype] = torch.int8
    standard_layout_columns: ClassVar[int] = 2
    requires_contiguous_experts: ClassVar[bool] = True
    prefer_hybrid_ep: ClassVar[bool] = True

    def __init__(self, config: MoEComputeConfig, selection):
        self.config = config
        self.selection = selection
        self.hidden = config.hidden
        self.intermediate = config.intermediate
        self.local_num_experts = config.local_num_experts
        self.global_num_experts = config.global_num_experts
        self.output_scale = config.output_scale
        self.limit = config.limit
        error = self.support_error(config, selection)
        if error is not None:
            raise NotImplementedError(error)

    @classmethod
    def support_error(cls, config, selection):
        layouts = cls.supported_layouts.get(selection.arch, ())
        if selection.name != cls.name or selection.layout not in layouts:
            return f"{cls.name} cannot consume {selection.layout} on {selection.arch}"
        spec = cls.weight_specs[selection.layout]
        if min(config.hidden, config.intermediate, config.local_num_experts) <= 0:
            return "Expert dimensions and local expert count must be positive"
        if config.hidden % spec.alignment or config.intermediate % spec.alignment:
            return f"{cls.name} requires {spec.alignment}-aligned dimensions"
        if config.apply_router_weight_on_input:
            return "Compute backends require output TopK weighting"
        if config.activation not in ("silu", "swiglu") or config.limit is None:
            return "Compute backends require clamped SwiGLU"
        return None

    @property
    def weight_spec(self) -> MoEWeightSpec:
        return self.weight_specs[self.selection.layout]

    def validate_weights(self, weights):
        if weights.spec != self.weight_spec:
            raise ValueError(
                f"Expected {self.weight_spec.format_id}, got {weights.spec.format_id}"
            )
        device = weights.w13_weight.device
        for prefix, n, k in (
            ("w13", 2 * self.intermediate, self.hidden),
            ("w2", self.hidden, self.intermediate),
        ):
            w = getattr(weights, prefix + "_weight")
            s = getattr(weights, prefix + "_weight_scale")
            expected_w, expected_s = self.weight_spec.shapes(
                self.local_num_experts, n, k
            )
            if (tuple(w.shape), tuple(s.shape)) != (expected_w, expected_s):
                raise ValueError(
                    f"{prefix}: expected weight {expected_w}, scale {expected_s}; got {tuple(w.shape)}, {tuple(s.shape)}"
                )
            if (
                w.dtype != self.weight_spec.weight_dtype
                or s.dtype != self.weight_spec.scale_dtype
            ):
                raise TypeError(
                    f"{prefix}: invalid weight/scale dtype for {self.weight_spec.format_id}"
                )
            if (
                w.device != device
                or s.device != device
                or not w.is_contiguous()
                or not s.is_contiguous()
            ):
                raise ValueError(
                    f"{prefix}: weights/scales must be contiguous on one device"
                )

    def workspace_spec(self, rows, dtype, *, batched=False):
        # rows is expanded Standard capacity (T * topk) or padded Batched rows.
        qdtype = self.activation_dtype
        return MoEWorkspaceSpec(
            {
                "layout": (
                    (
                        self.local_num_experts,
                        2 if batched else self.standard_layout_columns,
                    ),
                    torch.int32,
                ),
                "mapping": ((rows,), torch.int32),
                "input": ((rows, self.hidden), qdtype),
                "input_scale": ((rows, 1), torch.float32),
                "gemm1": ((rows, 2 * self.intermediate), dtype),
                "gemm2": ((rows, self.hidden), dtype),
            }
        )

    def local_range(self, expert_map):
        start = (
            self.config.ep_rank * self.local_num_experts
            if expert_map is not None
            else 0
        )
        return start, start + self.local_num_experts

    @staticmethod
    def empty_standard_output(x, output, residual):
        if output is None:
            output = torch.empty_like(x)
        if residual is None:
            return output.zero_()
        return output.copy_(residual)

    @abstractmethod
    def run_standard(
        self,
        x,
        weights,
        topk_weights,
        topk_ids,
        expert_map,
        *,
        workspace,
        output=None,
        residual=None,
        w13_gemm_mode=None,
    ):
        """Write/return weighted [T,H]; allocate only if output is None.

        The caller owns output. No collective or post-reduce output copy.
        """

    @abstractmethod
    def run_batched(self, x, weights, scale, counts, *, output, workspace):
        """Write unweighted [E,capacity,H]; communication combine owns TopK."""


@dataclass(frozen=True)
class MoEComputeSelection:
    name: str
    layout: str
    arch: str


_BACKENDS: dict[str, type[MoEComputeBackend]] = {}
_BUILTINS_LOADED = False


def register_moe_backend(cls):
    """Register a concrete compute implementation; duplicate names are errors."""
    if not issubclass(cls, MoEComputeBackend) or inspect.isabstract(cls):
        raise TypeError("A backend must implement MoEComputeBackend")
    if not cls.name or cls.name in _BACKENDS:
        raise ValueError(f"Duplicate/empty MoE backend name: {cls.name!r}")
    if not cls.supported_layouts or any(
        not layouts or any(layout not in cls.weight_specs for layout in layouts)
        for layouts in cls.supported_layouts.values()
    ):
        raise ValueError(f"Incomplete weight contracts for {cls.name}")
    cls.supported_layouts = MappingProxyType(dict(cls.supported_layouts))
    cls.weight_specs = MappingProxyType(dict(cls.weight_specs))
    _BACKENDS[cls.name] = cls
    return cls


def _load_builtin_backends():
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        # Explicit discovery avoids package scanning and import-order priorities.
        from . import ilu_ixinfer_w4a8, ilu_ixmm_fp4fp8, ilu_ixmm_w4a8  # noqa: F401

        _BUILTINS_LOADED = True


def select_moe_backend(arch, *, name="auto", layout="auto", config=None):
    """Select by compute capabilities only; no quantization/config dependencies."""
    _load_builtin_backends()
    if name != "auto" and name not in _BACKENDS:
        raise ValueError(f"Unknown MoE backend {name!r}")
    candidates = sorted(_BACKENDS.values(), key=lambda cls: (cls.priority, cls.name))
    reasons = []
    for cls in candidates:
        if name != "auto" and name != cls.name:
            continue
        layouts = cls.supported_layouts.get(arch, ())
        if not layouts or (layout != "auto" and layout not in layouts):
            reasons.append(f"{cls.name}: {arch}/{layout} unsupported")
            continue
        selection = MoEComputeSelection(
            cls.name, layouts[0] if layout == "auto" else layout, arch
        )
        error = cls.support_error(config, selection) if config is not None else None
        if error is None:
            return selection
        reasons.append(error)
    raise ValueError("No compatible MoE compute backend: " + "; ".join(reasons))


def get_moe_backend(selection):
    _load_builtin_backends()
    try:
        cls = _BACKENDS[selection.name]
    except KeyError:
        raise ValueError(f"Unknown MoE backend {selection.name!r}") from None
    if selection.layout not in cls.supported_layouts.get(selection.arch, ()):
        raise ValueError(f"Unsupported selection {selection}")
    return cls


def make_moe_compute(config, selection):
    return get_moe_backend(selection)(config, selection)

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import replace_parameter

logger = init_logger(__name__)


def align_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


class GemmInfo:
    def __init__(
        self,
        format_str: str,
        k: int,
        n: int,
        format_mapping: dict[str, int] | None = None,
    ):
        if format_mapping is None:
            format_mapping = {"NN": 0, "NT": 1, "TN": 2}
        assert format_str in format_mapping

        self.format_str = format_str
        self.format = format_mapping[format_str]
        self.k = k
        self.n = n
        self.pad_k = k
        self.pad_n = n

    def get_k(self):
        return self.k

    def get_pad_k(self):
        return self.pad_k

    def get_n(self):
        return self.n

    def get_pad_n(self):
        return self.pad_n

    def get_weight_shape(self):
        raise NotImplementedError("Subclasses must implement get_weight_shape().")

    def get_pad_weight_shape(self):
        raise NotImplementedError("Subclasses must implement get_pad_weight_shape().")


class W4A8GroupGemmInfo(GemmInfo):
    def __init__(
        self,
        format_str: str,
        k: int,
        n: int,
    ):
        """
        w4a8
        group gemm
        TN: K % 128 == 0 && K >= 256。如果groupsize = -1，则K >= 128
        NN: N % 128 == 0 && K % 64 == 0 && K >= 256

        group gemv
        TN: K % 128 == 0 && N % 16 == 0 (TODO: any N to support)
        NN: N % 128 == 0 && K % 64 == 0
        """
        super().__init__(format_str=format_str, k=k, n=n)

        if self.format_str == "TN":
            self.pad_k = align_up(k, 128)
            self.pad_k = max(256, self.pad_k)
            self.pad_n = align_up(n, 16)
        elif self.format_str == "NN":
            self.pad_k = align_up(k, 64)
            self.pad_n = align_up(n, 128)
            self.pad_k = max(256, self.pad_k)

        logger.info_once(
            f"[W4A8GroupGemmInfo] format: {format_str}, k: {k} -> {self.pad_k}, n: {n} -> {self.pad_n}"
        )

    # pack到int8  内层维度需要 / 2
    def get_pad_weight_shape(self):
        if self.format_str == "TN":
            # n, k
            return (self.pad_n, self.pad_k // 2)
        elif self.format_str == "NN":
            # k, n
            return (self.pad_k, self.pad_n // 2)

    def get_weight_shape(self):
        if self.format_str == "TN":
            # n, k
            return (self.n, self.k // 2)
        elif self.format_str == "NN":
            # k, n
            return (self.k, self.n // 2)


class W4A8GemmInfo(GemmInfo):
    def __init__(
        self,
        format_str: str,
        k: int,
        n: int,
    ):
        """
        w4a8
        gemm
        TN: Unsupported
        NN: N % 128 == 0 && K % 64 == 0 K >= 256

        gemv
        TN: Unsupported
        NN: N % 128 == 0 && K % 16 == 0
        """
        super().__init__(format_str=format_str, k=k, n=n)

        if self.format_str == "TN":
            raise AssertionError("W4A8 gemm with TN format is not supported")
        elif self.format_str == "NN":
            self.pad_k = align_up(k, 64)
            self.pad_k = max(256, self.pad_k)
            self.pad_n = align_up(n, 128)

        logger.info_once(
            f"[W4A8GemmInfo] format: {format_str}, k: {k} -> {self.pad_k}, n: {n} -> {self.pad_n}"
        )

    # pack到int8  内层维度需要 / 2
    def get_pad_weight_shape(self):
        if self.format_str == "TN":
            # n, k
            return (self.pad_n, self.pad_k // 2)
        elif self.format_str == "NN":
            # k, n
            return (self.pad_k, self.pad_n // 2)

    def get_weight_shape(self):
        if self.format_str == "TN":
            # n, k
            return (self.n, self.k // 2)
        elif self.format_str == "NN":
            # k, n
            return (self.k, self.n // 2)


class W8A8GemmInfo(GemmInfo):
    def __init__(
        self,
        format_str: str,
        k: int,
        n: int,
    ):
        """
        w8a8
        gemm
        TN: K % 64 == 0 && N % 2 == 0
        NN: N % 64 == 0 && K % 64 == 0

        gemv
        TN: K % 64 == 0
        NN: N % 64 == 0 && K % 64 == 0 (opt config : input_K % 64 == 0 && weight_K % 16 == 0, K use weight_K)
        """
        super().__init__(format_str=format_str, k=k, n=n)

        if self.format_str == "TN":
            self.pad_k = align_up(k, 64)
            self.pad_n = align_up(n, 2)
        elif self.format_str == "NN":
            self.pad_k = align_up(k, 64)
            self.pad_n = align_up(n, 64)

        logger.info_once(
            f"[W8A8GemmInfo] format: {format_str}, k: {k} -> {self.pad_k}, n: {n} -> {self.pad_n}"
        )

    def get_pad_weight_shape(self):
        if self.format_str == "TN":
            # n, k
            return (self.pad_n, self.pad_k)
        elif self.format_str == "NN":
            # k, n
            return (self.pad_k, self.pad_n)

    def get_weight_shape(self):
        if self.format_str == "TN":
            # n, k
            return (self.n, self.k)
        elif self.format_str == "NN":
            # k, n
            return (self.k, self.n)


class W8A8GroupGemmInfo(GemmInfo):
    """GEMM metadata for the W8A8 MoE group-gemm / group-gemv kernels (NN only).

    Mirrors W4A8GroupGemmInfo but for int8 weights (no int4 packing, so the
    stored weight shape is not halved).  Only the NN layout is supported:
    both moe_w8a8_group_gemm (cuinfer) and moe_w8a8_group_gemv accept NN, and
    moe_w8a8_group_gemv enforces NN exclusively
    (moe.cpp: `moe_w8a8_group_gemv only supported NN now!`).

    NN alignment (cuinfer W8A8, see ixformer src/ixformer/infer/w8a8.cpp):
      N % 64 == 0 && K % 64 == 0
    """

    def __init__(
        self,
        format_str: str,
        k: int,
        n: int,
    ):
        super().__init__(format_str=format_str, k=k, n=n)
        assert self.format_str == "NN", (
            f"W8A8GroupGemmInfo only supports NN, got {self.format_str}"
        )

        self.pad_k = align_up(k, 64)
        self.pad_n = align_up(n, 64)

        logger.info_once(
            f"[W8A8GroupGemmInfo] format: {format_str}, "
            f"k: {k} -> {self.pad_k}, n: {n} -> {self.pad_n}"
        )

    # int8 (no packing): the stored weight shape equals the logical shape.
    # NN layout stores weight as [k, n].
    def get_pad_weight_shape(self):
        return (self.pad_k, self.pad_n)

    def get_weight_shape(self):
        return (self.k, self.n)


def hidden_padding(real_k: int, pad_k: int) -> int:
    """Extra hidden elements needed to reach kernel-aligned pad_k."""
    assert pad_k >= real_k, f"pad_k ({pad_k}) must be >= real_k ({real_k})"
    return pad_k - real_k


def pad_tensor(input: torch.Tensor, pad_dim: int):
    # tokens, hidden_size
    assert input.dim() == 2
    assert input.shape[-1] <= pad_dim
    if input.shape[-1] == pad_dim:
        return input

    padding = pad_dim - input.shape[-1]
    return torch.nn.functional.pad(input, (0, padding), mode="constant", value=0)


def unpad_tensor(input: torch.Tensor, real_dim: int, contiguous: bool = False):
    # tokens, hidden_size
    assert input.dim() == 2
    assert real_dim <= input.shape[-1]
    if input.shape[-1] == real_dim:
        return input

    if contiguous:
        return input[..., :real_dim].contiguous()
    else:
        return input[..., :real_dim]


def replace_with_padded(
    layer: torch.nn.Module,
    name: str,
    target_shape: tuple[int, ...],
    pad_value: int | float = 0,
    extra_attrs: dict[str, object] | None = None,
    log_prefix: str | None = None,
) -> None:
    """Pad an existing parameter to target_shape and replace it in module.

    Existing public attributes on the original parameter are preserved on the
    new parameter. `extra_attrs` can be used to explicitly override/add attributes.
    """
    if pad_value not in (0, 1):
        raise ValueError(
            f"replace_with_padded only supports pad_value 0 or 1, got {pad_value}"
        )

    param = getattr(layer, name, None)
    if param is None:
        return

    if tuple(param.shape) == target_shape:
        return

    old_attrs = {k: v for k, v in vars(param).items() if not k.startswith("_")}
    if len(target_shape) != param.dim():
        raise ValueError(
            f"Mismatched rank for {name}: {param.dim()} -> {len(target_shape)}"
        )
    for src_dim, dst_dim in zip(param.shape, target_shape):
        if dst_dim < src_dim:
            raise ValueError(
                f"Unexpected shape shrink for {name}: "
                f"{tuple(param.shape)} -> {target_shape}"
            )

    padded = (
        torch.zeros(target_shape, dtype=param.dtype, device=param.device)
        if pad_value == 0
        else torch.ones(target_shape, dtype=param.dtype, device=param.device)
    )
    copy_slices = tuple(slice(0, dim) for dim in param.shape)
    padded[copy_slices].copy_(param.data)

    if log_prefix is not None:
        logger.info_once(
            f"{log_prefix} {name}: shape {tuple(param.shape)} -> {target_shape}, "
            f"dtype {param.dtype} -> {padded.dtype}, "
        )

    replace_parameter(
        layer,
        name,
        torch.nn.Parameter(padded.contiguous(), requires_grad=False),
    )

    new_param = getattr(layer, name)
    for k, v in old_attrs.items():
        setattr(new_param, k, v)
    if extra_attrs is not None:
        for k, v in extra_attrs.items():
            setattr(new_param, k, v)

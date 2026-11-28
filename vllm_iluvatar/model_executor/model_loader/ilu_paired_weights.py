# SPDX-License-Identifier: Apache-2.0
"""Checkpoint source descriptions routed through existing model weight loaders.

Only view operations are deferred. Ordinary data operations materialize their
inputs; MXFP4 callbacks retain descriptions until the complete plan is checked.
"""

from collections import OrderedDict
from dataclasses import dataclass
from os.path import abspath

import torch
from safetensors import safe_open
from torch.utils._pytree import tree_map

_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "U16": torch.uint16,
    "I32": torch.int32,
    "U32": torch.uint32,
    "I64": torch.int64,
    "U64": torch.uint64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E8M0": torch.float8_e8m0fnu,
}


@dataclass(frozen=True)
class TensorSource:
    file: str
    name: str


@dataclass(frozen=True)
class TensorReference:
    source: TensorSource
    metadata: torch.Tensor
    views: tuple = ()

    def read(self, readers):
        tensor = readers.read(self.source)
        for op, args, kwargs in self.views:
            tensor = op(tensor, *args, **kwargs)
        return tensor


class DeferredCheckpointTensor(torch.Tensor):
    """Storage-free CPU tensor preserving a checkpoint source through views.

    This is not a fake execution of model.load_weights. That function runs once,
    with normal writes for ordinary parameters. Data operations fall through to
    real tensors. A streaming MXFP4 callback rejects sources that have already
    undergone an unsupported data transformation instead of retaining payloads.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl
    _views = frozenset(
        (
            torch.ops.aten.view.default,
            torch.ops.aten.view.dtype,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.select.int,
            torch.ops.aten.detach.default,
            torch.ops.aten.transpose.int,
            torch.ops.aten.permute.default,
            torch.ops.aten.alias.default,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
            torch.ops.aten.squeeze.dims,
        )
    )

    @staticmethod
    def __new__(cls, reference, readers):
        meta = reference.metadata
        obj = torch.Tensor._make_wrapper_subclass(
            cls,
            meta.shape,
            strides=meta.stride(),
            storage_offset=meta.storage_offset(),
            dtype=meta.dtype,
            device="cpu",
            requires_grad=False,
        )
        obj.reference = reference
        obj.readers = readers
        return obj

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.split.Tensor and isinstance(args[0], cls):
            # ``Tensor.chunk`` lands here, and a model that stores a layer's
            # experts as one fused tensor splits gate from up with it (Hy4's
            # ``experts.gate_up_proj``). Each piece is a slice, which replays
            # through the single-output path below; the last one is short when
            # the dimension does not divide evenly.
            value, size = args[0], args[1]
            dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
            length = value.reference.metadata.shape[dim]
            return tuple(
                torch.ops.aten.slice.Tensor(
                    value, dim, begin, min(begin + size, length)
                )
                for begin in range(0, length, size)
            )
        # ``RoutedExperts.load_weights`` unbinds a fused [E, ...] shard into
        # per-expert rows (MiniMax-M2.5 block-FP8). Without this, unbind
        # materializes and paired loading rejects the plain Tensor.
        if func is torch.ops.aten.unbind.int and isinstance(args[0], cls):
            value = args[0]
            dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
            length = value.reference.metadata.shape[dim]
            return tuple(
                torch.ops.aten.select.int(value, dim, index) for index in range(length)
            )
        if func in cls._views and isinstance(args[0], cls):
            value = args[0]
            ref = value.reference
            meta = func(ref.metadata, *args[1:], **kwargs)
            return cls(
                TensorReference(
                    ref.source, meta, ref.views + ((func, args[1:], kwargs),)
                ),
                value.readers,
            )

        def materialize(value):
            if isinstance(value, cls):
                return value.reference.read(value.readers)
            return value

        return func(*tree_map(materialize, args), **tree_map(materialize, kwargs))


def checkpoint_references(files, *, weight_map=None):
    """Read headers and use the index to resolve sources for indexed tensors.

    Some exporters omit auxiliary scales from the index. Still load those
    from the selected shards, rejecting duplicates without an indexed source.
    """
    references = []
    if weight_map is not None:
        weight_map = {name: abspath(file) for name, file in weight_map.items()}
        selected_files = {abspath(file) for file in files}
        indexed_names = {
            name for name, file in weight_map.items() if file in selected_files
        }
    seen = set()
    for file in files:
        path = abspath(file)
        with safe_open(file, framework="pt", device="cpu") as reader:
            for name in reader.keys():  # noqa: SIM118
                if weight_map is not None and weight_map.get(name, path) != path:
                    continue
                if name in seen:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                seen.add(name)
                view = reader.get_slice(name)
                dtype = _DTYPES.get(view.get_dtype())
                if dtype is None:
                    raise ValueError(
                        f"Unsupported checkpoint dtype: {name} {view.get_dtype()}"
                    )
                meta = torch.empty(view.get_shape(), dtype=dtype, device="meta")
                references.append(TensorReference(TensorSource(file, name), meta))
    if weight_map is not None:
        missing = indexed_names - seen
        if missing:
            name = min(missing)
            raise ValueError(
                f"Indexed checkpoint tensor {name} missing from {weight_map[name]}"
            )
    return references


class SafetensorsReaders:
    """Bounded file-handle cache; tensors are consumed before the next read."""

    def __init__(self, capacity=16):
        if capacity < 1:
            raise ValueError("Reader capacity must be positive")
        self.capacity = capacity
        self.handles = OrderedDict()
        self.opens = 0
        self.reads = 0
        self.closed = False

    def read(self, source):
        if self.closed:
            raise RuntimeError("Checkpoint reader session is closed")
        self.reads += 1
        if source.file not in self.handles:
            if len(self.handles) == self.capacity:
                _, (context, _) = self.handles.popitem(last=False)
                context.__exit__(None, None, None)
            context = safe_open(source.file, framework="pt", device="cpu")
            reader = context.__enter__()
            self.handles[source.file] = (context, reader)
            self.opens += 1
        self.handles.move_to_end(source.file)
        return self.handles[source.file][1].get_tensor(source.name)

    def close(self):
        self.closed = True
        errors = []
        while self.handles:
            _, (context, _) = self.handles.popitem()
            try:
                context.__exit__(None, None, None)
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]

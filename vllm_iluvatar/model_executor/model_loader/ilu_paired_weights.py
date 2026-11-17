# SPDX-License-Identifier: Apache-2.0
"""Checkpoint source descriptions routed through existing model weight loaders.

Only view operations are deferred. Ordinary data operations materialize their
inputs; MXFP4 callbacks retain descriptions until the complete plan is checked.
"""

from collections import OrderedDict
from dataclasses import dataclass

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


def checkpoint_references(files):
    """Read headers only; reject duplicate names before yielding any sources."""
    references = []
    seen = set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as reader:
            for name in reader.keys():  # noqa: SIM118
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

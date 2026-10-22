"""Typed CPU tensor storage with exact BF16 bytes and checked offsets."""

from __future__ import annotations

import math

import torch

DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.int64,
        torch.int32,
        torch.uint8,
    )
}


def pack_tensors(values):
    fields = []
    parts = []
    offset = 0
    for name, tensor in sorted(values.items()):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or str(tensor.dtype) not in DTYPES
        ):
            raise ValueError("Expected a supported CPU encoder-input tensor")
        tensor = tensor.contiguous()
        size = tensor.numel() * tensor.element_size()
        aligned = (
            (offset + tensor.element_size() - 1)
            // tensor.element_size()
            * tensor.element_size()
        )
        parts.append(bytes(aligned - offset))
        offset = aligned
        fields.append(
            dict(
                name=name,
                dtype=str(tensor.dtype),
                shape=list(tensor.shape),
                offset=offset,
                bytes=size,
            )
        )
        parts.append(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        offset += size
    return fields, b"".join(parts)


def unpack_tensors(fields, raw):
    values = {}
    end = 0
    for f in fields:
        dtype = DTYPES[f["dtype"]]
        shape = f["shape"]
        offset = f["offset"]
        size = f["bytes"]
        element = torch.empty((), dtype=dtype).element_size()
        if (
            not isinstance(f["name"], str)
            or f["name"] in values
            or not isinstance(shape, list)
            or any(type(x) is not int or x <= 0 for x in shape)
            or type(offset) is not int
            or type(size) is not int
            or offset < end
            or offset % element
            or size != math.prod(shape) * element
            or offset + size > len(raw)
        ):
            raise ValueError("Invalid CPU tensor storage layout")
        values[f["name"]] = torch.frombuffer(
            bytearray(raw[offset : offset + size]), dtype=dtype
        ).reshape(shape)
        end = offset + size
    if end != len(raw):
        raise ValueError("Unexpected encoder-input storage suffix")
    return values

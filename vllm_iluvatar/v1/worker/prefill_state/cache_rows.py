"""CPU-sized views for the CR1 boundary row not covered by even KV replay."""

import torch


def indexer_row_views(cache: torch.Tensor, block: int, row: int,
                      scales: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
    """Return actual key and scale byte ranges, without a device allocation.

    Quantized indexer pages contain all128-byte keys followed by all4-byte
    scales. Their nominal [rows,132] view is not an array of132-byte records.
    BF16 pages contain ordinary contiguous128-element rows.
    """
    if not 0 <= row < cache.shape[1] or not 0 <= block < cache.shape[0]:
        raise ValueError("Indexer row extends beyond its current page")
    if scales is not None:
        if (cache.dtype != torch.uint8 or cache.shape[-1] != 128
                or scales.dtype != torch.float32 or scales.shape != cache.shape[:2]
                or scales.device != cache.device or scales.stride(1) != 1):
            raise ValueError("Unsupported E25 split indexer boundary-row layout")
        return cache[block, row], scales[block, row:row + 1].view(torch.uint8)
    if cache.dtype == torch.uint8 and cache.shape[-1] == 132:
        rows = cache.shape[1]
        page = cache[block].as_strided((rows * 132,), (1,))
        return (page[row * 128:(row + 1) * 128],
                page[rows * 128 + row * 4:rows * 128 + (row + 1) * 4])
    if cache.dtype == torch.bfloat16 and cache.shape[-1] == 128:
        return (cache[block, row].view(torch.uint8),)
    raise ValueError("Unsupported E25 indexer boundary-row layout")


def copy_indexer_row_to_host(host: torch.Tensor, cache: torch.Tensor,
                             block: int, row: int,
                             scales: torch.Tensor | None = None) -> None:
    parts = indexer_row_views(cache, block, row, scales)
    if host.dtype != torch.uint8 or host.numel() != sum(p.numel() for p in parts):
        raise ValueError("Indexer boundary payload size mismatch")
    offset = 0
    for part in parts:
        host[offset:offset + part.numel()].copy_(part, non_blocking=True)
        offset += part.numel()


def restore_indexer_row(cache: torch.Tensor, block: int, row: int,
                        host: torch.Tensor,
                        scales: torch.Tensor | None = None) -> None:
    parts = indexer_row_views(cache, block, row, scales)
    if host.dtype != torch.uint8 or host.numel() != sum(p.numel() for p in parts):
        raise ValueError("Indexer boundary payload size mismatch")
    offset = 0
    for part in parts:
        part.copy_(host[offset:offset + part.numel()], non_blocking=True)
        offset += part.numel()



def main_boundary_sources(layer, name: str, group: int, geometry) -> list[dict]:
    """Describe paired live page views, including padded page strides.

    Payload and scale planes use the same physical block and row. This creates
    metadata only and allocates no device workspace.
    """
    dtypes = {"BF16": torch.bfloat16, "U8": torch.uint8, "FP16": torch.float16}
    payload = layer.kv_cache
    sources = []
    for kind, (shape, dtype) in geometry.main_tail_fields().items():
        cache = payload if kind == "main_tail" else getattr(layer, "kv_cache_scales", None)
        if (not isinstance(cache, torch.Tensor) or cache.ndim != 3
                or cache.shape[:2] != payload.shape[:2] or cache.shape[-1] != shape[-1]
                or cache.dtype != dtypes[dtype] or cache.device != payload.device
                or cache.stride(-1) != 1):
            raise ValueError("Unsupported E25 paired main boundary-row layout")
        sources.append(dict(name=name if kind == "main_tail" else name + ".kv_cache_scales",
                            kind=kind, layer=geometry.prefix_depth - 1, group=group,
                            cache=cache, R=int(cache.shape[1]), ratio=1))
    return sources

"""CPU-sized views for the CR1 boundary row not covered by even KV replay."""

import torch


def indexer_row_views(cache: torch.Tensor, block: int, row: int) -> tuple[torch.Tensor, ...]:
    """Return actual key and scale byte ranges, without a device allocation.

    Quantized indexer pages contain all128-byte keys followed by all4-byte
    scales. Their nominal [rows,132] view is not an array of132-byte records.
    BF16 pages contain ordinary contiguous128-element rows.
    """
    if not 0 <= row < cache.shape[1] or not 0 <= block < cache.shape[0]:
        raise ValueError("Indexer row extends beyond its current page")
    if cache.dtype == torch.uint8 and cache.shape[-1] == 132:
        rows = cache.shape[1]
        page = cache[block].as_strided((rows * 132,), (1,))
        return (page[row * 128:(row + 1) * 128],
                page[rows * 128 + row * 4:rows * 128 + (row + 1) * 4])
    if cache.dtype == torch.bfloat16 and cache.shape[-1] == 128:
        return (cache[block, row].view(torch.uint8),)
    raise ValueError("Unsupported E25 indexer boundary-row layout")


def copy_indexer_row_to_host(host: torch.Tensor, cache: torch.Tensor,
                             block: int, row: int) -> None:
    parts = indexer_row_views(cache, block, row)
    if host.dtype != torch.uint8 or host.numel() != sum(p.numel() for p in parts):
        raise ValueError("Indexer boundary payload size mismatch")
    offset = 0
    for part in parts:
        host[offset:offset + part.numel()].copy_(part, non_blocking=True)
        offset += part.numel()


def restore_indexer_row(cache: torch.Tensor, block: int, row: int,
                        host: torch.Tensor) -> None:
    parts = indexer_row_views(cache, block, row)
    if host.dtype != torch.uint8 or host.numel() != sum(p.numel() for p in parts):
        raise ValueError("Indexer boundary payload size mismatch")
    offset = 0
    for part in parts:
        part.copy_(host[offset:offset + part.numel()], non_blocking=True)
        offset += part.numel()

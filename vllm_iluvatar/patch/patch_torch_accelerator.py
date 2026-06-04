"""Backfill ``torch.accelerator`` for older iluvatar corex PyTorch builds.

The internal fork patches this in ``vllm/__init__.py`` (executed at
``import vllm`` time). vLLM only *calls* ``torch.accelerator.*`` at runtime
(worker memory management), which is well after the plugin loads, so applying
the same shim from ``adapt_patch`` is sufficient and avoids needing a source
edit / ``.pth`` bootstrap.
"""

from __future__ import annotations

_PATCHED = False


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import types

    import torch

    if not hasattr(torch, "accelerator"):
        torch.accelerator = types.ModuleType("torch.accelerator")  # type: ignore

    accel = torch.accelerator
    patch_map = {
        "empty_cache": torch.cuda.empty_cache,
        "device_count": torch.cuda.device_count,
        "set_device_index": torch.cuda.set_device,
        "current_device_index": torch.cuda.current_device,
        "memory_stats": torch.cuda.memory_stats,
        "memory_reserved": torch.cuda.memory_reserved,
        "max_memory_allocated": torch.cuda.max_memory_allocated,
        "reset_peak_memory_stats": torch.cuda.reset_peak_memory_stats,
        "synchronize": torch.cuda.synchronize,
        "device_index": torch.cuda.device,
    }
    for name, func in patch_map.items():
        if not hasattr(accel, name):
            setattr(accel, name, func)

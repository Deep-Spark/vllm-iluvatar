"""Reclaim dead process sessions without touching another live engine."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .archive import atomic_json

OWNER_KIND = "vllm_iluvatar_mm_replay_v1"


def owner_identity():
    fields = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    return dict(
        kind=OWNER_KIND,
        pid=os.getpid(),
        start_ticks=int(fields[19]),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    )


def owner_alive(owner):
    if (
        owner.get("boot_id")
        != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    ):
        return False
    try:
        fields = (
            Path(f"/proc/{int(owner['pid'])}/stat")
            .read_text()
            .rsplit(") ", 1)[1]
            .split()
        )
    except (FileNotFoundError, ProcessLookupError):
        return False
    return fields[0] != "Z" and int(fields[19]) == owner.get("start_ticks")


def claim_session(root):
    """Only reclaim our explicitly marked dead sessions; unknown paths stay intact."""
    root = Path(root)
    for sibling in root.parent.iterdir():
        if (
            not sibling.is_dir()
            or sibling.is_symlink()
            or sibling == root
            or len(sibling.name) != 64
            or any(c not in "0123456789abcdef" for c in sibling.name)
        ):
            continue
        marker = sibling / "owner.json"
        try:
            if not marker.is_file():
                continue
            owner = json.loads(marker.read_text())
            if owner.get("kind") != OWNER_KIND or owner_alive(owner):
                continue
        except (OSError, ValueError, KeyError):
            continue
        shutil.rmtree(sibling, ignore_errors=True)
    atomic_json(root / "owner.json", owner_identity())

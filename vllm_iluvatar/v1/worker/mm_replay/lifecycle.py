"""Reclaim dead process sessions without touching another live engine."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .archive import atomic_json

# Old collectors understand only v1 and must leave namespace-aware sessions
# alone during a rolling upgrade. This changes local housekeeping, not the
# image transport protocol or the scheduler/worker directory identity.
OWNER_KIND = "vllm_iluvatar_mm_replay_v2"


def owner_identity():
    fields = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    return dict(
        kind=OWNER_KIND,
        pid=os.getpid(),
        start_ticks=int(fields[19]),
        boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        pid_namespace=os.readlink("/proc/self/ns/pid"),
    )


def owner_alive(owner):
    """Keep an owner unless its death can be established in our PID namespace.

    Separate containers can share /dev/shm while their PID numbers refer to
    unrelated processes. Missing or mismatched /proc entries are meaningful
    only after the namespace is known to match. Unscoped legacy records and
    inaccessible process identities are conservatively retained.
    """
    if owner.get("pid_namespace") != os.readlink("/proc/self/ns/pid"):
        return True
    # A bind-mounted /proc can expose an ancestor's PID numbering even when
    # our own PID namespace matches the marker. Do not probe that view either.
    if os.readlink("/proc/self") != str(os.getpid()):
        return True
    if (
        type(owner.get("pid")) is not int
        or owner["pid"] <= 0
        or type(owner.get("start_ticks")) is not int
        or owner["start_ticks"] < 0
        or not isinstance(owner.get("boot_id"), str)
        or not owner["boot_id"]
    ):
        return True
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
    except OSError:
        return True
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
            if (
                not isinstance(owner, dict)
                or owner.get("kind") != OWNER_KIND
                or owner_alive(owner)
            ):
                continue
        except (OSError, ValueError, KeyError):
            continue
        shutil.rmtree(sibling, ignore_errors=True)
    atomic_json(root / "owner.json", owner_identity())

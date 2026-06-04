#!/usr/bin/env python3
"""Install (or remove) the interpreter-startup ``.pth`` bootstrap.

The bootstrap makes the mirror-tree override finder active BEFORE ``import
vllm`` so that early-imported modules -- most importantly
``vllm.distributed.parallel_state`` and the device communicators -- resolve to
the iluvatar overlay on their very first import. This is what lets the plugin
keep the ixformer communication optimization (single module object, no custom-op
double-registration, no split global state).

It writes a ``.pth`` file into the active environment's site-packages whose
single executable line prepends this plugin's source dir to ``sys.path`` and
imports :mod:`vllm_iluvatar._bootstrap`::

    import sys; sys.path.insert(0, '<plugin_src>'); import vllm_iluvatar._bootstrap

Embedding the source path makes it robust for editable installs (we don't rely
on ``.pth`` processing order vs the editable-install ``.pth``).

Usage::

    python scripts/install_bootstrap.py            # install
    python scripts/install_bootstrap.py --uninstall # remove
    python scripts/install_bootstrap.py --print     # show what would be written

WITHOUT this bootstrap the plugin still loads via the ``vllm.general_plugins``
entry point, but the distributed comm stack falls back to community vLLM (NCCL),
i.e. the ixformer comm optimization is disabled.
"""

from __future__ import annotations

import argparse
import sysconfig
from pathlib import Path

_PTH_NAME = "vllm_iluvatar_bootstrap.pth"


def _plugin_src() -> Path:
    # scripts/install_bootstrap.py -> repo root (parent of the vllm_iluvatar pkg).
    return Path(__file__).resolve().parent.parent


def _site_packages() -> Path:
    return Path(sysconfig.get_path("purelib"))


def _pth_line(plugin_src: Path) -> str:
    # Single physical line; site.py executes lines starting with "import ".
    return (
        f"import sys; sys.path.insert(0, {str(plugin_src)!r}); "
        f"import vllm_iluvatar._bootstrap\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uninstall", action="store_true", help="remove the .pth bootstrap"
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="print the .pth target/content without writing",
    )
    args = parser.parse_args()

    plugin_src = _plugin_src()
    sp = _site_packages()
    pth_path = sp / _PTH_NAME
    line = _pth_line(plugin_src)

    if args.print_only:
        print(f"target : {pth_path}")
        print(f"content: {line}", end="")
        return 0

    if args.uninstall:
        if pth_path.exists():
            pth_path.unlink()
            print(f"[OK] removed {pth_path}")
        else:
            print(f"[--] not present: {pth_path}")
        return 0

    # Sanity: make sure the package is importable from plugin_src.
    if not (plugin_src / "vllm_iluvatar" / "_bootstrap.py").is_file():
        print(
            f"[FAIL] {plugin_src}/vllm_iluvatar/_bootstrap.py not found; "
            "run this from the plugin checkout."
        )
        return 1

    try:
        pth_path.write_text(line, encoding="utf-8")
    except OSError as e:
        print(f"[FAIL] could not write {pth_path}: {e}")
        print("       (need write access to site-packages; try a venv or sudo)")
        return 1

    print(f"[OK] wrote {pth_path}")
    print(f"     -> {line}", end="")
    print(
        "\nNew Python processes in this environment now install the override "
        "finder at startup.\nVerify:\n"
        '  python -c "import sys; '
        "print(any(type(f).__name__=='_IluRedirectFinder' for f in sys.meta_path))\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

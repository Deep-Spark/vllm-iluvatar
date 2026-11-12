# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remove the "developer preview" announcement banner on tagged releases.

On Read the Docs, a tagged (released) version should not show the banner that
tells readers they are viewing the development preview. The banner itself lives
in ``overrides/main.html`` (the ``announce`` block).

How
---
The banner is stripped from the **built HTML** after the page is generated,
by dropping the element that Material emits for the announce block:

    <div data-md-component="announce"><aside class="md-banner">…</aside></div>

Why not delete the source file
------------------------------
An earlier version of this hook did ``os.remove("overrides/main.html")``. That
mutates the working tree during a build: running

    READTHEDOCS_VERSION_TYPE=tag mkdocs build

locally silently deleted a tracked file, and the build still "succeeded" (the
banner just disappeared). Read the Docs happens to use a fresh checkout per
build, so it never noticed — but a single local invocation destroys the file.
This hook therefore never touches the source tree.
"""

import os
import re
from pathlib import Path
from typing import Literal

# <div data-md-component=announce> … </div> wrapping the banner.
# Material may emit attribute values unquoted (minified output), so the opening
# tag is matched loosely. The announce div contains no nested <div> of its own,
# so a non-greedy match up to the matching close tag is safe.
_ANNOUNCE_RE = re.compile(
    r'<div\s+data-md-component\s*=\s*["\']?announce["\']?\s*>.*?</div>',
    re.IGNORECASE | re.DOTALL,
)

# Set by Read the Docs: <https://docs.readthedocs.io/en/stable/reference/environment-variables.html>
_VERSION_TYPE = "READTHEDOCS_VERSION_TYPE"


def _strip_announce(html: str) -> str:
    return _ANNOUNCE_RE.sub("", html, count=1)


def on_post_page(output: str, page, config, **kwargs) -> str:
    """Drop the announce banner from a rendered page on tagged builds."""
    if os.getenv(_VERSION_TYPE) != "tag":
        return output
    return _strip_announce(output)


def on_post_build(config, **kwargs) -> None:
    """Also sweep files MkDocs renders outside the page pipeline.

    ``404.html`` comes from Material's own template, not from a Markdown page,
    so ``on_post_page`` never sees it. Sweep the built site directory instead.
    Only files under ``site_dir`` are touched — the source tree is never read
    for writing.
    """
    if os.getenv(_VERSION_TYPE) != "tag":
        return

    site_dir = Path(config.site_dir)
    if not site_dir.is_dir():
        return

    swept = 0
    for html_path in site_dir.rglob("*.html"):
        text = html_path.read_text(encoding="utf-8", errors="replace")
        if "data-md-component" not in text or "announce" not in text:
            continue
        new = _strip_announce(text)
        if new != text:
            html_path.write_text(new, encoding="utf-8")
            swept += 1
    if swept:
        print(f"[remove_announcement] swept banner from {swept} extra page(s)")


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    """Report the intended behaviour up front, so builds are self-explanatory."""
    if os.getenv(_VERSION_TYPE) == "tag":
        print("[remove_announcement] tagged build: stripping the preview banner "
              "from rendered pages (source files untouched)")

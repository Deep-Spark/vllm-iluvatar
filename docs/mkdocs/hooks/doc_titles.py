# SPDX-License-Identifier: Apache-2.0
"""Give developer docs a real nav/browser title without touching their body.

Some developer guides open with an SPDX HTML comment, so MkDocs'
`get_markdown_title` (which only accepts a first non-empty line of "# ")
falls back to the file name. That shows up as "4 how to use ..." in the
sidebar and the browser tab.

The body is never modified: only the Page title is filled from the document's
own H1, and only when MkDocs fell back to the file name.
"""

import re
from pathlib import Path

from mkdocs.utils import get_markdown_title
from mkdocs.structure.pages import Page

H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def _fill(page: Page) -> None:
    src = page.file.content_string
    if get_markdown_title(src):
        return  # MkDocs found a real H1 itself; leave it alone.
    match = H1_RE.search(src)
    if match:
        page.title = match.group(1)


def on_nav(nav, *, config, files):
    for page in nav.pages:
        _fill(page)
    return nav

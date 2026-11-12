# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Show one entry per release and keep shared guides out of the sidebar.

Release notes live under ``release/<version>/`` as a ``README.md`` wrapper plus
language variants (``en.md`` / ``zh.md``). This hook collapses each such folder
into a single navigation entry (so the sidebar shows one row per version, with
the language switch kept on the page) and removes the shared ``release/common``
section from the sidebar while keeping its pages reachable.

Adapted from the upstream vLLM / vLLM-MetaX docs hook.
"""

from pathlib import PurePosixPath

from mkdocs.structure.nav import Navigation
from mkdocs.structure.pages import Page


def on_nav(nav: Navigation, **kwargs) -> Navigation:
    def collapse(items):
        for position, item in enumerate(items):
            if not item.is_section:
                continue

            pages = item.children
            if not all(isinstance(child, Page) for child in pages):
                collapse(pages)
                continue

            index = next((child for child in pages if child.is_index), None)
            if index is None:
                continue
            directory = PurePosixPath(index.file.src_uri).parent
            if directory.parent != PurePosixPath("release"):
                continue
            if any(
                PurePosixPath(child.file.src_uri).parent != directory for child in pages
            ):
                continue
            names = {PurePosixPath(child.file.src_uri).name for child in pages}
            if names not in (
                {"README.md", "en.md", "zh.md"},
                {"README.md", "common_en.md", "common_zh.md"},
            ):
                continue

            for child in pages:
                child.title = item.title
                child.parent = item.parent

            # Material's nav-item macro selects the current page from `pages`.
            index.pages = pages  # type: ignore[attr-defined]
            items[position] = index

        # Keep common pages reachable without a separate sidebar row.
        items[:] = [
            item
            for item in items
            if not (
                isinstance(item, Page)
                and PurePosixPath(item.file.src_uri).parent
                == PurePosixPath("release/common")
            )
        ]

    collapse(nav.items)
    return nav

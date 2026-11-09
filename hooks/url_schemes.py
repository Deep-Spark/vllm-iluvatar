# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MkDocs hook to enable the following links to render correctly:

- Relative file links outside of the `docs/` directory, e.g.:
    - [Text](../some_file.py)
    - [Directory](../../some_directory/)
  are rewritten to point at the file/directory on GitHub.
- GitHub URLs for issues, pull requests, and projects, e.g.:
    - Adds a GitHub icon before links
    - Replaces raw links with descriptive text,
        e.g. <...pull/123> -> [Pull Request #123](.../pull/123)
    - Works for external repos too by including the `owner/repo` in the title

The goal is to simplify cross-referencing common GitHub resources in the docs.

Adapted from the upstream vLLM docs hook for vllm-iluvatar
(GitHub mirror: Deep-Spark/vllm-iluvatar).
"""

from pathlib import Path

import regex as re
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page

# hooks/ lives at the repo root, so the repo root is one level up.
ROOT_DIR = Path(__file__).parent.parent.resolve()
DOC_DIR = ROOT_DIR / "docs"

# The branch that rewritten source links should point at. It must match the
# branch used by `edit_uri` in mkdocs.yaml, otherwise "edit this page" and the
# source links disagree and one of them 404s. This used to be hard-coded to
# `main`, but the Deep-Spark mirror has no `main` branch (its default is
# release/vX.Y.Z), so every rewritten link was broken.
#
# Derived from `edit_uri` at runtime so the two can never drift apart. Set
# `extra.repo_web.ref` only to override that, e.g. for a preview build.
GH_REPO = "Deep-Spark/vllm-iluvatar"
GH_ICON = ":octicons-mark-github-16:"

DEFAULT_REF = "release/v0.27.1"

DEFAULT_REPO_WEB = {
    "repo": GH_REPO,
    "base": f"https://github.com/{GH_REPO}",
}


def _ref_from_config(config: MkDocsConfig, repo_web: dict) -> str:
    """Pick the branch for rewritten links.

    `extra.repo_web.ref` wins when set; otherwise take the ref from
    `edit_uri` (``edit/<ref>/<docs_dir>/``) so "edit this page" and the source
    links can never point at different branches.
    """
    if repo_web.get("ref"):
        return repo_web["ref"]
    edit_uri = (config.get("edit_uri") or "").strip()
    parts = [p for p in edit_uri.split("/") if p]
    if len(parts) >= 3 and parts[0] == "edit":
        return "/".join(parts[1:-1]) or DEFAULT_REF
    return DEFAULT_REF


# Regex pieces
TITLE = r"(?P<title>[^\[\]<>]+?)"
REPO = r"(?P<repo>.+?/.+?)"
TYPE = r"(?P<type>issues|pull|projects)"
NUMBER = r"(?P<number>\d+)"
FRAGMENT = r"(?P<fragment>#[^\s]+)?"
URL = f"https://github.com/{REPO}/{TYPE}/{NUMBER}{FRAGMENT}"
RELATIVE = r"(?!(https?|ftp)://|#)(?P<path>[^\s]+?)"

# Common titles to use for GitHub links when none is provided in the link.
TITLES = {"issues": "Issue ", "pull": "Pull Request ", "projects": "Project "}

github_link = re.compile(rf"(\[{TITLE}\]\(|<){URL}(\)|>)")
relative_link = re.compile(rf"\[{TITLE}\]\({RELATIVE}\)")


def on_page_markdown(
    markdown: str, *, page: Page, config: MkDocsConfig, files: Files
) -> str:
    repo_web = {**DEFAULT_REPO_WEB, **(config.get("extra") or {}).get("repo_web", {})}
    base = repo_web["base"].rstrip("/")
    ref = _ref_from_config(config, repo_web)
    # `exclude_docs` is a GitIgnoreSpec: files it matches are still present in
    # `files`, but they never produce HTML, so a relative link to one is broken.
    exclude = config.get("exclude_docs")

    def _is_built(src_uri: str) -> bool:
        return not (exclude and exclude.match_file(src_uri))

    def replace_relative_link(match: re.Match) -> str:
        """Point relative links at the repository when the target is not in the site.

        Two cases need this: a path outside ``docs/`` (a source file), and a
        docs page dropped by ``exclude_docs`` (e.g. ``developer_guide/guides/``).
        Both would otherwise render as a broken site link, so send them to the
        repository instead. Anything that *is* built stays a normal relative link.
        """
        title = match.group("title")
        path = match.group("path")
        path = (Path(page.file.abs_src_path).parent / path).resolve()

        if not path.exists():
            return match.group(0)

        if path.is_relative_to(DOC_DIR):
            src_uri = path.relative_to(DOC_DIR).as_posix()
            if path.is_dir():
                # A directory is only reachable on the site through its index.
                if _is_built(f"{src_uri.rstrip('/')}/index.md"):
                    return match.group(0)
            elif _is_built(src_uri):
                return match.group(0)

        rel = path.relative_to(ROOT_DIR).as_posix()
        kind = "tree" if path.is_dir() else "blob"
        url = f"{base}/{kind}/{ref}/{rel}"
        return f"[{GH_ICON} {title}]({url})"

    def replace_github_link(match: re.Match) -> str:
        """Enhance GitHub issue, PR, and project links."""
        repo = match.group("repo")
        type = match.group("type")
        number = match.group("number")
        title = match.group("title") or ""
        fragment = match.group("fragment") or ""

        if not title:
            title = TITLES[type]
            if GH_REPO.split("/")[0] not in repo:
                title += repo
            title += f"#{number}"

        url = f"https://github.com/{repo}/{type}/{number}{fragment}"
        return f"[{GH_ICON} {title}]({url})"

    markdown = relative_link.sub(replace_relative_link, markdown)
    markdown = github_link.sub(replace_github_link, markdown)
    return markdown

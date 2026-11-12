"""Render or remove <!-- doc-sources --> markers.

Default builds (including the public Read the Docs build) delete the marker
entirely, so the published site never shows the evidence blocks.

Set DOCS_EDITION=preview (local preview only) to lay each section out side by
side: body on the left, its evidence table on the right. The layout CSS is
injected into preview pages only, so the formal site carries no trace of it.
"""

import os
import re

BLOCK_RE = re.compile(r"[ \t]*<!--\s*doc-sources\b(.*?)-->[ \t]*", re.DOTALL)
ASIDE_RE = re.compile(r'<div class="ds-aside">.*?</div>\s*', re.DOTALL)
NEXT_HEADING_RE = re.compile(r"<h[1-6][\s>]")

PREVIEW_CSS = """<style>
.md-grid{max-width:96rem}
.md-content{max-width:none}
.ds-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1.2rem;
  align-items:start;margin:.4rem 0 1.2rem;border-top:1px dashed var(--md-default-fg-color--lightest);padding-top:.6rem}
.ds-row>.ds-body>:first-child{margin-top:0}
.ds-aside{position:sticky;top:5.2rem;font-size:.62rem;background:var(--md-code-bg-color);
  border-left:3px solid #00bfa5;border-radius:4px;padding:.4rem .7rem;max-height:80vh;overflow:auto}
.ds-aside::before{content:"依据 - preview 版本显示，正式版删除";display:block;font-weight:700;color:#00897b;margin-bottom:.3rem}
.ds-aside table{font-size:.6rem}
.ds-aside .md-typeset__table{margin:0}
@media (max-width:76em){.ds-row{grid-template-columns:1fr}.ds-aside{position:static}}
</style>
"""


def _preview_markdown(body):
    lines = [line.rstrip() for line in body.splitlines() if line.strip()]
    if not lines:
        return ""
    inner = "\n".join(lines)
    return f'\n\n<div class="ds-aside" markdown="1">\n\n{inner}\n\n</div>\n\n'


def on_page_markdown(markdown, **kwargs):
    preview = os.environ.get("DOCS_EDITION") == "preview"

    def replace(match):
        if not preview:
            return ""
        return _preview_markdown(match.group(1))

    return BLOCK_RE.sub(replace, markdown)


def on_page_content(html, **kwargs):
    if os.environ.get("DOCS_EDITION") != "preview" or 'class="ds-aside"' not in html:
        return html

    out, pos = [], 0
    for m in ASIDE_RE.finditer(html):
        if m.start() < pos:
            continue
        out.append(html[pos:m.start()])
        nxt = NEXT_HEADING_RE.search(html, m.end())
        end = nxt.start() if nxt else len(html)
        aside = m.group(0).strip()
        body = html[m.end():end]
        out.append(f'<div class="ds-row"><div class="ds-body">{body}</div>{aside}</div>')
        pos = end
    out.append(html[pos:])
    return PREVIEW_CSS + "".join(out)

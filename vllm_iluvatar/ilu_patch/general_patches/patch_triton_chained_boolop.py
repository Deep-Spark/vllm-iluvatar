# SPDX-License-Identifier: Apache-2.0
"""Adapt the vLLM penalties kernel to the CoreX Triton parser.

Corresponds to: vllm/v1/worker/gpu/sample/penalties.py
Target   : vllm.v1.worker.gpu.sample.penalties._penalties_kernel (JIT ``.src``)
Baseline : community iluvatar_main a04c2a2b62
Why      : CoreX 4.5's Triton parser rejects boolean chains with three
           operands. The public V2 runner compiles ``_penalties_kernel``
           during sampler warmup, before any request can be served.
How      : Rewrite only that JIT function's source, folding ``A or B or C``
           into ``(A or B) or C``. No community file is modified.
Scope    : CoreX Triton sampler warmup on Iluvatar. Does not participate in
           Qwen GDN / QSA / PLE / HC / mamba-align math.
Remove   : when CoreX Triton accepts chained boolean operators, or when
           upstream splits the penalties kernel into pairwise ops.
"""

from __future__ import annotations

import ast
import re

from vllm_iluvatar.ilu_patch import general_patch as p

_TARGET_MODULE = "vllm.v1.worker.gpu.sample.penalties"
_TARGET_ATTR = "_penalties_kernel"
_ILU_REWRITTEN_ATTR = "_ilu_split_chained_boolops"


class _SplitBoolOps(ast.NodeTransformer):
    """Fold ``A or B or C`` into ``(A or B) or C``."""

    def __init__(self) -> None:
        self.count = 0

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.BoolOp:
        self.generic_visit(node)
        if len(node.values) <= 2:
            return node
        self.count += 1
        folded = node.values[0]
        for value in node.values[1:]:
            folded = ast.BoolOp(op=node.op, values=[folded, value])
        return ast.copy_location(ast.fix_missing_locations(folded), node)


def _rewrite_kernel(jit_fn: object) -> int:
    if getattr(jit_fn, _ILU_REWRITTEN_ATTR, False):
        return 0
    src = getattr(jit_fn, "src", None)
    if not isinstance(src, str):
        return 0

    transformer = _SplitBoolOps()
    tree = transformer.visit(ast.parse(src))
    if not transformer.count:
        setattr(jit_fn, _ILU_REWRITTEN_ATTR, True)
        return 0

    new_src = ast.unparse(ast.fix_missing_locations(tree))
    match = re.search(r"^def\s+\w+\s*\(", new_src, re.MULTILINE)
    if match is None:
        raise RuntimeError("Could not locate the penalties Triton function body")
    jit_fn.src = new_src[match.start() :]
    jit_fn.hash = None
    setattr(jit_fn, _ILU_REWRITTEN_ATTR, True)
    return transformer.count


def _apply() -> None:
    import importlib

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    module = importlib.import_module(_TARGET_MODULE)
    jit_fn = getattr(module, _TARGET_ATTR, None)
    if jit_fn is None:
        return
    count = _rewrite_kernel(jit_fn)
    if count:
        logger.debug(
            "ILU: split %d chained boolean operator(s) in %s.%s",
            count,
            _TARGET_MODULE,
            _TARGET_ATTR,
        )


def register() -> None:
    p.register_once(
        __name__,
        _apply,
        detail="adapt the vLLM penalties kernel to the CoreX Triton parser",
    )

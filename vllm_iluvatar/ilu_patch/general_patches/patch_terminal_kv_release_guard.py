# SPDX-License-Identifier: Apache-2.0
"""Keep cancelled asynchronous Multi KV receives until both owners finish.

This experimental plugin patch preserves the connector's logical completion
contract. It neither queries device events nor makes failed transfers safe for
DMA. The original scheduler still performs the actual, step-fenced block free.
"""
from __future__ import annotations

import __future__
import ast
from copy import copy
from dataclasses import dataclass
from functools import wraps
import os
from pathlib import Path
import textwrap
from types import CodeType
from typing import Any

FLAG = "VLLM_ILUVATAR_TERMINAL_KV_OWNERSHIP"
_MARK = "_ilu_terminal_kv_ownership_installed"
_STATE = "_ilu_terminal_kv_ownership"


@dataclass
class _Ownership:
    request: Any
    send_required: bool | None = None
    recv_done: bool = False
    send_done: bool = False
    admitting: bool = True
    free_attempted: bool = False


def _records(scheduler: Any) -> dict[str, _Ownership]:
    return scheduler.__dict__.setdefault(_STATE, {})


def _eligible(scheduler: Any, delay_free_blocks: bool) -> bool:
    connector = getattr(scheduler, "connector", None)
    # EC has a separate completion contract. Keep that configuration unchanged.
    return (
        delay_free_blocks is True
        and getattr(scheduler, "ec_connector", None) is None
        and len(getattr(connector, "_connectors", ())) > 1
        and getattr(connector.update_connector_output, "_ilu_multi_completion_sources", False)
    )


def _consume_completed(scheduler: Any, output: Any) -> set[str]:
    records = _records(scheduler)
    sends = set(output.finished_sending or ())
    recvs = set(output.finished_recving or ())
    handled = (sends | recvs) & records.keys()
    ready = []
    for request_id in handled:
        owner = records[request_id]
        if scheduler.requests.get(request_id) is not owner.request:
            raise RuntimeError("Terminal KV request identity changed: " + request_id)
        if owner.admitting or owner.send_required is None:
            raise RuntimeError("Terminal KV send responsibility is unknown: " + request_id)
        if owner.free_attempted:
            # A failed allocator/free may have partially changed ownership.
            # Propagate the failure; never retry it as a second free.
            raise RuntimeError("Terminal KV free was already attempted: " + request_id)
        if request_id in sends and owner.send_done:
            raise RuntimeError("Duplicate final KV send completion: " + request_id)
        if request_id in recvs and owner.recv_done:
            raise RuntimeError("Duplicate final KV receive completion: " + request_id)
        if request_id in sends:
            if not owner.send_required:
                raise RuntimeError("Unexpected terminal KV send completion: " + request_id)
            owner.send_done = True
        if request_id in recvs:
            # Only the connector's final R signal counts. A failed_recving flag
            # alone is not R; failed/invalid metadata itself remains untouched.
            owner.recv_done = True
        if owner.recv_done and (not owner.send_required or owner.send_done):
            ready.append((request_id, owner))
    for request_id, owner in ready:
        owner.free_attempted = True
        scheduler._free_blocks(owner.request)
        if scheduler.requests.get(request_id) is owner.request:
            raise RuntimeError("Original KV free did not retire request: " + request_id)
        del records[request_id]
    return set(handled)


def _code_signature(code: CodeType) -> tuple:
    """Compare executable bodies without their source filename/line numbers."""
    constants = tuple(_code_signature(v) if isinstance(v, CodeType) else v for v in code.co_consts)
    return (
        code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
        code.co_nlocals, code.co_stacksize, code.co_flags, code.co_code,
        code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars,
        getattr(code, "co_exceptiontable", b""), constants,
    )


def _verify_consumer_body(consumer: Any, transform_source: Any) -> None:
    source_file = consumer.__globals__.get("__file__")
    if not source_file:
        raise RuntimeError("KV consumer has no original scheduler source file")
    source = Path(source_file).read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "Scheduler"
               for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name == "_update_from_kv_xfer_finished"]
    if len(methods) != 1:
        raise RuntimeError("Original scheduler KV consumer is missing or ambiguous")
    method = methods[0]
    if method.decorator_list:
        raise RuntimeError("Original scheduler KV consumer decorators need review")
    original_source = textwrap.dedent("\n".join(source.splitlines()[method.lineno - 1:method.end_lineno]))
    patched = transform_source(original_source)
    compiled = compile(patched, "<terminal-kv-expected>", "exec",
                       flags=__future__.annotations.compiler_flag, dont_inherit=True)
    bodies = [v for v in compiled.co_consts if isinstance(v, CodeType)
              and v.co_name == "_update_from_kv_xfer_finished"]
    if len(bodies) != 1 or _code_signature(consumer.__code__) != _code_signature(bodies[0]):
        raise RuntimeError("KV consumer differs from the original cc02-transformed body")


def install_terminal_kv_release_guard(scheduler_cls: type, *, transform_source=None) -> bool:
    """Wrap the current, already reordered scheduler methods once.

    Capturing the current functions preserves the prefill-budget wrapper. The
    old KV consumer must already have had its late connector update removed by
    patch_kv_load_failure_recovery; the outer update remains its sole caller.
    """
    if getattr(scheduler_cls, _MARK, False):
        return False
    old_free = scheduler_cls._free_request
    old_connector_finished = scheduler_cls._connector_finished
    old_consumer = scheduler_cls._update_from_kv_xfer_finished
    if not getattr(old_consumer, "_ilu_kv_load_recovery_patched", False):
        raise RuntimeError("Terminal KV guard requires reordered KV output handling")
    if not getattr(scheduler_cls.update_from_output, "_ilu_kv_load_recovery_patched", False):
        raise RuntimeError("Terminal KV guard requires the original early connector update")
    if "update_connector_output" in old_consumer.__code__.co_names:
        raise RuntimeError("Terminal KV consumer still updates its connector")
    if transform_source is None:
        from vllm_iluvatar.ilu_patch.general_patches.patch_kv_load_failure_recovery import (
            patch_kv_xfer_finished_source,
        )
        transform_source = patch_kv_xfer_finished_source
    _verify_consumer_body(old_consumer, transform_source)

    @wraps(old_free)
    def free_request(self, request, delay_free_blocks=False):
        if not _eligible(self, delay_free_blocks):
            return old_free(self, request, delay_free_blocks=delay_free_blocks)
        assert request.is_finished()
        request_id = request.request_id
        if self.requests.get(request_id) is not request:
            raise RuntimeError("Terminal KV admission has no current request: " + request_id)
        records = _records(self)
        if request_id in records:
            raise RuntimeError("Terminal KV ownership was already admitted: " + request_id)
        owner = _Ownership(request)
        records[request_id] = owner
        try:
            result = old_free(self, request, delay_free_blocks=delay_free_blocks)
        finally:
            owner.admitting = False
        if owner.send_required is None:
            raise RuntimeError("Original connector did not report KV ownership: " + request_id)
        return result

    @wraps(old_connector_finished)
    def connector_finished(self, request):
        result = old_connector_finished(self, request)
        owner = _records(self).get(request.request_id)
        if owner is not None and owner.admitting and owner.request is request:
            if not isinstance(result, tuple) or len(result) != 2 or type(result[0]) is not bool:
                raise RuntimeError("Original connector returned unknown KV ownership")
            if owner.send_required is not None:
                raise RuntimeError("Original connector reported KV ownership twice")
            owner.send_required = result[0]
        return result

    @wraps(old_consumer)
    def consume(self, output):
        handled = _consume_completed(self, output)
        if not handled:
            return old_consumer(self, output)
        # Preserve the caller's output, metadata, failures, stats and events.
        # Only the old parent's release loops receive filtered completion sets.
        view = copy(output)
        view.finished_sending = (
            None if output.finished_sending is None else set(output.finished_sending) - handled
        )
        view.finished_recving = (
            None if output.finished_recving is None else set(output.finished_recving) - handled
        )
        return old_consumer(self, view)

    scheduler_cls._free_request = free_request
    scheduler_cls._connector_finished = connector_finished
    scheduler_cls._update_from_kv_xfer_finished = consume
    setattr(scheduler_cls, _MARK, True)
    return True


def register() -> None:
    """Register the default-enabled guard after its dependencies have run."""
    if os.environ.get(FLAG, "1") != "1":
        return
    from vllm_iluvatar.ilu_patch import general_patch as patches
    from vllm_iluvatar.ilu_patch.general_patches import (
        patch_kv_load_failure_recovery,
        patch_prefill_budget,
    )

    patch_kv_load_failure_recovery.register()
    patch_prefill_budget.register()
    patches.register_once(__name__, _apply, detail="terminal asynchronous Multi KV ownership")


def _apply() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.logger import init_logger

    if install_terminal_kv_release_guard(Scheduler):
        init_logger(__name__).info("Installed terminal Multi KV ownership guard")

"""Per-rank weight snapshot, so an out-of-bounds kernel can be caught after the fact.

``IluDeepseekV41Model`` starts one of these in every worker. Each DP/TP/PP rank
binds its own port and serves a single endpoint:

    POST /v1/chat/save_weights   {"tag": "<name>"}
    GET  /v1/chat/save_weights

A tag is written once, to ``/data/nlp/dsv41_weight_dump/<tag>/dp<D>_pp<P>_tp<T>/``.
A second call with the same tag is refused, so a rank can only be snapshotted
twice by using two tags. Engram parameters are skipped (they live on CPU, and
are not what an attention kernel could scribble on); so is anything else that
is not a CUDA parameter.

Stdlib HTTP rather than FastAPI: a worker process already has its own threads
and this must not spin up a second ASGI loop.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from torch import nn

from vllm.distributed import get_dp_group, get_pp_group, get_tp_group
from vllm.logger import init_logger

logger = init_logger(__name__)

# 19000 + dp * DP_STRIDE + pp * PP_STRIDE + tp.
# PP_STRIDE stays above any tp world we run; DP_STRIDE covers pp*PP_STRIDE
# for pp up to 16 so DP ranks do not collide on the same host.
_PORT_BASE = 19000
_PP_STRIDE = 16
_DP_STRIDE = 256
_DUMP_ROOT = Path("/data/nlp/dsv41_weight_dump")
_TAG_OK = re.compile(r"[A-Za-z0-9_-]{1,32}")
_ENGRAM = re.compile(r"(^|\.)engram\.")


def port_for(dp_rank: int, pp_rank: int, tp_rank: int) -> int:
    return _PORT_BASE + dp_rank * _DP_STRIDE + pp_rank * _PP_STRIDE + tp_rank


def _sha256(cpu: torch.Tensor) -> str:
    """Hash raw bytes. Chunked so a multi-GiB tensor is never copied whole."""
    cpu = cpu.contiguous()
    nbytes = cpu.numel() * cpu.element_size()
    if nbytes == 0:
        return hashlib.sha256(b"").hexdigest()
    ptr = cpu.untyped_storage().data_ptr()
    hasher = hashlib.sha256()
    step = 64 * 1024 * 1024
    for off in range(0, nbytes, step):
        n = min(step, nbytes - off)
        hasher.update((ctypes.c_char * n).from_address(ptr + off))
    return hasher.hexdigest()


def dump_weights(
    model: nn.Module, tag: str, dp_rank: int, pp_rank: int, tp_rank: int
) -> dict:
    # The listener starts at the end of ``__init__``, which is before
    # ``load_weights``. Snapshotting then would record uninitialised storage.
    if not getattr(model, "_weights_ready", False):
        raise RuntimeError("weights are not loaded yet")
    if not _TAG_OK.fullmatch(tag):
        raise ValueError(f"tag must match {_TAG_OK.pattern}")
    dest = _DUMP_ROOT / tag / f"dp{dp_rank}_pp{pp_rank}_tp{tp_rank}"
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; a tag is written once")

    # The request is served on a thread that has never called set_device, so
    # its current device is not this worker's GPU. Pin it before synchronising,
    # or the sync hits the wrong device and reports whatever fault is pending
    # there.
    device = next(
        (p.device for _, p in model.named_parameters() if p.is_cuda),
        None,
    )
    if device is not None:
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)

    dest.mkdir(parents=True)

    manifest = []
    skipped = []
    total_bytes = 0
    for name, param in model.named_parameters():
        if _ENGRAM.search(name) or param.device.type == "cpu":
            skipped.append(
                {"name": name, "device": str(param.device), "dtype": str(param.dtype)}
            )
            continue
        cpu = param.detach().to("cpu")
        digest = _sha256(cpu)
        fname = name.replace("/", "_") + ".pt"
        torch.save(cpu, dest / fname)
        nbytes = cpu.numel() * cpu.element_size()
        total_bytes += nbytes
        manifest.append(
            {
                "name": name,
                "file": fname,
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "nbytes": nbytes,
                "sha256": digest,
            }
        )
        del cpu

    summary = {
        "tag": tag,
        "dp": dp_rank,
        "pp": pp_rank,
        "tp": tp_rank,
        "num_tensors": len(manifest),
        "num_skipped": len(skipped),
        "total_bytes": total_bytes,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest))
    (dest / "skipped.json").write_text(json.dumps(skipped))
    # Written last: its presence is what a client waits on.
    (dest / "DONE").write_text(json.dumps(summary))
    logger.info(
        "weight dump %s dp=%s pp=%s tp=%s tensors=%s bytes=%.2fGiB skipped=%s",
        tag,
        dp_rank,
        pp_rank,
        tp_rank,
        len(manifest),
        total_bytes / (1 << 30),
        len(skipped),
    )
    return summary


def start_weight_dump_server(model: nn.Module) -> None:
    """Bind this worker's port. A failure to bind is logged, not fatal."""
    dp_rank = get_dp_group().rank_in_group
    pp_rank = get_pp_group().rank_in_group
    tp_rank = get_tp_group().rank_in_group
    port = port_for(dp_rank, pp_rank, tp_rank)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?")[0] != "/v1/chat/save_weights":
                self.send_error(404)
                return
            self._send(
                {
                    "dp": dp_rank,
                    "pp": pp_rank,
                    "tp": tp_rank,
                    "port": port,
                    "ready": bool(getattr(model, "_weights_ready", False)),
                    "root": str(_DUMP_ROOT),
                }
            )

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?")[0] != "/v1/chat/save_weights":
                self.send_error(404)
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(n) or b"{}")
                tag = str(body["tag"])
                with lock:
                    summary = dump_weights(model, tag, dp_rank, pp_rank, tp_rank)
            except FileExistsError as exc:
                self._send({"error": str(exc)}, status=409)
                return
            except Exception as exc:  # reported to the caller, also logged
                logger.exception("weight dump failed")
                self._send({"error": f"{type(exc).__name__}: {exc}"}, status=500)
                return
            self._send(summary)

        def _send(self, payload: dict, status: int = 200) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt: str, *args) -> None:
            logger.info("weight dump http: " + fmt, *args)

    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    except OSError:
        logger.exception("weight dump failed to bind port %s", port)
        return
    httpd.daemon_threads = True
    thread = threading.Thread(
        target=httpd.serve_forever,
        name=f"weight-dump-dp{dp_rank}-pp{pp_rank}-tp{tp_rank}",
        daemon=True,
    )
    thread.start()
    logger.info(
        "weight dump listening on :%s (dp=%s pp=%s tp=%s)",
        port,
        dp_rank,
        pp_rank,
        tp_rank,
    )

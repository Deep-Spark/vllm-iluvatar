# SPDX-License-Identifier: Apache-2.0
"""Opt-in worker: observe real steps and save bounded replay inputs.

Only selected through --worker-cls by the diagnostic runner. Production and
performance runs use the normal IluWorker and pay no observation overhead.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from vllm_iluvatar.backends import provider_mode

if provider_mode() == "ixaite":
    from vllm_iluvatar.backends.ixaite.worker import IxaiteWorker as SelectedWorker
else:
    from vllm_iluvatar.v1.worker.gpu_worker import IluWorker as SelectedWorker


# Copy tensors to the replay artifact outside the profiled production op range.
def _cpu(x):
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x


# Wrap only the first instance of each op per real prefill step, on each TP rank.
class ChunkedPrefillWorker(SelectedWorker):
    def load_model(self, **kwargs):
        super().load_model(**kwargs)
        self._audit_dir = Path(os.environ["SWCOMP3266_AUDIT_DIR"]) / f"rank{self.rank}"
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._audit_step = 0
        self._audit_active = False
        self._audit_seen = set()
        self._audit_traced = False
        self._audit_prompts = {}
        self._audit_computed = {}
        # Community GDN has no strided vendor ABI; capture the own ABI only.
        if provider_mode() == "ixaite":
            from vllm_iluvatar.backends.ixaite.attention import flash_attn as fa
            from vllm_iluvatar.backends.ixaite.layers import gdn

            self._audit_gdn = gdn._GDN_CHUNK
            gdn._GDN_CHUNK = self._capture_gdn
            name = "_SELECTED_FLASH_ATTN_OPS"
        else:
            from vllm_iluvatar.v1.attention.backends import ilu_flash_attn as fa

            name = "_IXFORMER_FLASH_ATTN_OPS"
        selected = getattr(fa, name)
        self._audit_fa = selected[0]
        setattr(fa, name, (self._capture_fa, *selected[1:]))

    # Read the scheduler's actual request progress, not guessed query lengths.
    def execute_model(self, scheduler_output):
        for req in scheduler_output.scheduled_new_reqs:
            self._audit_prompts[req.req_id] = len(req.prompt_token_ids)
            self._audit_computed[req.req_id] = req.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        self._audit_computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        counts = scheduler_output.num_scheduled_tokens
        requests = [
            dict(
                request_id=r,
                query_tokens=n,
                computed=self._audit_computed[r],
                prompt_tokens=self._audit_prompts[r],
            )
            for r, n in counts.items()
        ]
        prefills = sum(r["computed"] < r["prompt_tokens"] for r in requests)
        decodes = len(requests) - prefills
        row = dict(
            step=self._audit_step, prefills=prefills, decodes=decodes, requests=requests
        )
        self._audit_seen = set()
        self._audit_active = prefills > 0
        # One mixed step is sufficient to identify kernels; avoid profiling all
        # 48 GDN layers on every decode and contaminating the performance run.
        continuation = any(0 < r["computed"] < r["prompt_tokens"] for r in requests)
        trace = continuation and decodes > 0 and not self._audit_traced
        row["profiled_continuation"] = trace
        try:
            if trace:
                with torch.profiler.profile(
                    record_shapes=True,
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                ) as prof:
                    result = super().execute_model(scheduler_output)
                prof.export_chrome_trace(str(self._audit_dir / "mixed-trace.json"))
                self._audit_traced = True
            else:
                result = super().execute_model(scheduler_output)
        finally:
            self._audit_active = False
        with (self._audit_dir / "steps.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        for r, n in counts.items():
            self._audit_computed[r] += n
        for r in scheduler_output.finished_req_ids:
            self._audit_prompts.pop(r, None)
            self._audit_computed.pop(r, None)
        self._audit_step += 1
        return result

    # Compact the selected state rows for offline baseline replay; the actual
    # provider still receives the original pool and indices unchanged.
    def _capture_gdn(self, *args):
        capture = self._audit_active and "gdn" not in self._audit_seen
        if capture:
            self._audit_seen.add("gdn")
            (
                q,
                k,
                v,
                out,
                g,
                beta,
                state,
                mask,
                cu,
                cu_cpu,
                indices,
                scale,
                output_indices,
            ) = args
            rows = indices.long()
            saved = dict(
                q=_cpu(q),
                k=_cpu(k),
                v=_cpu(v),
                g=_cpu(g),
                beta=_cpu(beta),
                state=_cpu(state.index_select(0, rows)),
                mask=_cpu(mask),
                cu=_cpu(cu),
                indices=_cpu(indices),
                state_stride=state.stride(),
                scale=scale,
            )
        with torch.profiler.record_function("SWCOMP3266_GDN_PRODUCTION"):
            result = self._audit_gdn(*args)
        if capture:
            saved["out"] = _cpu(
                out if output_indices is None else out[output_indices.long()]
            )
            saved["final_state"] = _cpu(state.index_select(0, rows))
            torch.save(saved, self._audit_dir / f"{self._audit_step:04d}-gdn.pt")
        return result

    # Keep only pages used by this step, remapping their IDs in the saved copy.
    def _capture_fa(self, **kwargs):
        capture = self._audit_active and "fa" not in self._audit_seen
        if capture:
            self._audit_seen.add("fa")
            table = kwargs["block_table"]
            pages, inverse = torch.unique(table, return_inverse=True)
            saved = {
                k: _cpu(v)
                for k, v in kwargs.items()
                if k not in ("k", "v", "out", "block_table")
            }
            saved.update(
                k=_cpu(kwargs["k"][pages.long()]),
                v=_cpu(kwargs["v"][pages.long()]),
                block_table=_cpu(inverse.reshape_as(table).to(table.dtype)),
            )
        with torch.profiler.record_function("SWCOMP3266_FA_PRODUCTION"):
            result = self._audit_fa(**kwargs)
        if capture:
            torch.save(
                dict(kwargs=saved, out=_cpu(kwargs["out"])),
                self._audit_dir / f"{self._audit_step:04d}-fa.pt",
            )
        return result

# SPDX-License-Identifier: Apache-2.0
"""Opt-in capture of real all-mode states and cascade calls."""

import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch

from vllm_iluvatar.backends import provider_mode

if provider_mode() == "ixaite":
    from vllm_iluvatar.backends.ixaite.worker import IxaiteWorker as SelectedWorker
else:
    from vllm_iluvatar.v1.worker.gpu_worker import IluWorker as SelectedWorker


# Capture tensors only in diagnostic runs, outside measured provider scopes.
def cpu(value):
    return value.detach().cpu() if isinstance(value, torch.Tensor) else value


class PrefixCacheWorker(SelectedWorker):
    """Observe scheduler hits and capture bounded inputs on both TP ranks."""

    def load_model(self, **kwargs):
        """Install wrappers after providers and model construction have completed."""
        super().load_model(**kwargs)
        self.directory = Path(os.environ["SWCOMP3265_AUDIT_DIR"]) / f"rank{self.rank}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.active = False
        self.seen = set()
        self.cascade_traced = False
        self.gdn_traced = False
        self.chunk = None
        if provider_mode() == "ixaite":
            from vllm_iluvatar.backends.ixaite.attention import flash_attn as fa
            from vllm_iluvatar.backends.ixaite.layers import gdn

            self.chunk = gdn._IXAITE_APC_CHUNK
            gdn._IXAITE_APC_CHUNK = self.capture_chunk
        else:
            from vllm_iluvatar.v1.attention.backends import ilu_flash_attn as fa

        self.cascade = fa.cascade_attention
        fa.cascade_attention = self.capture_cascade

    def execute_model(self, scheduler_output):
        """Record the scheduler's actual initial cache-hit token counts."""
        row = dict(
            step=self.step,
            common_prefix_blocks=scheduler_output.num_common_prefix_blocks,
            scheduled=scheduler_output.num_scheduled_tokens,
            new=[
                dict(
                    id=r.req_id,
                    computed=r.num_computed_tokens,
                    prompt_tokens=len(r.prompt_token_ids),
                )
                for r in scheduler_output.scheduled_new_reqs
            ],
        )
        self.seen = set()
        self.active = True
        try:
            return super().execute_model(scheduler_output)
        finally:
            self.active = False
            with (self.directory / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            self.step += 1

    def capture_chunk(self, q, k, v, g, beta, **kw):
        """Keep shared physical rows shared when compacting the replay pool."""
        if not self.active or "gdn" in self.seen:
            return self.chunk(q, k, v, g, beta, **kw)
        self.seen.add("gdn")
        table = kw["state_block_table"]
        computed = kw["num_computed_tokens_cpu"]
        cu = kw["cu_seqlens_cpu"]
        block_size = kw["cache_block_size"]
        # Include only source and scheduled columns; future allocation columns
        # are irrelevant and can hold null slots.
        compact_table = table.cpu().clone()
        sources = kw["initial_state_indices"].cpu()
        slots = set(sources.tolist())
        for i in range(len(computed)):
            start = int(computed[i]) // block_size
            end = (int(computed[i]) + int(cu[i + 1] - cu[i]) - 1) // block_size
            slots.update(compact_table[i, start : end + 1].tolist())
        rows = sorted(slots)
        mapping = {slot: i for i, slot in enumerate(rows)}
        for slot, index in mapping.items():
            compact_table[table.cpu() == slot] = index
        row_tensor = torch.tensor(rows, device=q.device, dtype=torch.long)
        saved = dict(
            q=cpu(q),
            k=cpu(k),
            v=cpu(v),
            g=cpu(g),
            beta=cpu(beta),
            state=cpu(kw["initial_state"].index_select(0, row_tensor)),
            table=compact_table,
            source=torch.tensor(
                [mapping[s] for s in sources.tolist()], dtype=torch.int32
            ),
            computed=cpu(computed),
            cu=cpu(cu),
            mask=cpu(kw["has_initial_state"]),
            block_size=block_size,
        )
        trace = not self.gdn_traced and bool((computed > 0).any())
        context = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            )
            if trace
            else nullcontext()
        )
        with context as prof, torch.profiler.record_function("swcomp3265_ixaite_gdn"):
            result = self.chunk(q, k, v, g, beta, **kw)
        if trace:
            prof.export_chrome_trace(str(self.directory / "gdn-trace.json"))
            self.gdn_traced = True
        saved["out"] = cpu(kw["out"])
        saved["final_state"] = cpu(kw["initial_state"].index_select(0, row_tensor))
        torch.save(saved, self.directory / f"{self.step:05d}-gdn.pt")
        return result

    def capture_cascade(self, output, query, key_cache, value_cache, **kw):
        """Capture one actual cascade per rank, including prefix/suffix/merge kernels."""
        if not self.active or self.cascade_traced:
            return self.cascade(output, query, key_cache, value_cache, **kw)
        table = kw["block_table"]
        blocks, inverse = torch.unique(table, return_inverse=True)
        saved = dict(
            query=cpu(query),
            key=cpu(key_cache[blocks.long()]),
            value=cpu(value_cache[blocks.long()]),
            kwargs={
                name: cpu(value)
                for name, value in kw.items()
                if name not in ("varlen_op", "merge_op")
            },
        )
        saved["kwargs"]["block_table"] = cpu(inverse.reshape_as(table).to(torch.int32))
        with (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof,
            torch.profiler.record_function("swcomp3265_ixaite_cascade"),
        ):
            result = self.cascade(output, query, key_cache, value_cache, **kw)
        prof.export_chrome_trace(str(self.directory / "cascade-trace.json"))
        saved["out"] = cpu(output)
        torch.save(saved, self.directory / "cascade.pt")
        self.cascade_traced = True
        return result

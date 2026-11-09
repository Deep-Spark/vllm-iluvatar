# SPDX-License-Identifier: Apache-2.0
"""embedding_gather cross-provider suite.

Baseline is the community PyTorch gather (``F.embedding``, what
``VocabParallelEmbedding.forward`` calls today); the FlyDSL side is the
ixaite op bound through the plugin provider. Both share the same
``(ids, weight) -> emb`` ABI, so one workload drives both. The op is a
pure row copy, so correctness is bit-exact and the aux metric is the
effective gather bandwidth.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tests.ixaite.benchmarks.core.model_catalog import l2_spec
from tests.ixaite.benchmarks.core.registry import register
from tests.ixaite.benchmarks.core.types import Providers, Workload
from tests.ixaite.benchmarks.suites._common import filter_cases


@dataclass(frozen=True)
class EmbeddingGatherCase:
    case_id: str
    num_tokens: int
    vocab_size: int
    hidden_size: int
    tags: tuple[str, ...] = ("embedding", "decode", "prefill")

    def make_workload(self, device: torch.device, dtype: torch.dtype) -> Workload:
        gen = torch.Generator(device=device).manual_seed(self.num_tokens)
        weight = torch.randn(
            (self.vocab_size, self.hidden_size),
            device=device,
            dtype=dtype,
            generator=gen,
        )
        ids = torch.randint(
            0,
            self.vocab_size,
            (self.num_tokens,),
            device=device,
            dtype=torch.int64,
            generator=gen,
        )
        return Workload(
            kwargs={"ids": ids, "weight": weight},
            meta={"T": self.num_tokens, "V": self.vocab_size, "H": self.hidden_size},
        )

    def run(self, fn: Callable[..., object], wl: Workload) -> None:
        wl.outputs["emb"] = fn(wl.kwargs["ids"], wl.kwargs["weight"])  # type: ignore[assignment]

    def check(self, baseline_wl: Workload, flydsl_wl: Workload) -> None:
        # A pure gather must reproduce F.embedding bit-exactly.
        torch.testing.assert_close(
            flydsl_wl.outputs["emb"],
            baseline_wl.outputs["emb"],
            rtol=0,
            atol=0,
        )

    def aux_from_ms(self, median_ms: float, wl: Workload) -> tuple[float, str] | None:
        # Effective bandwidth: one gathered row read plus one row write.
        elem = wl.kwargs["weight"].element_size()
        num_bytes = 2 * self.num_tokens * self.hidden_size * elem
        return num_bytes / (median_ms * 1e-3) / 1e9, "GB/s"


class EmbeddingGatherSuite:
    name = "embedding_gather"

    def resolve_providers(self) -> Providers:
        import ixaite  # noqa: F401 -- initialize FlyDSL before vllm imports it

        from vllm_iluvatar.backends.ixaite.runtime import load_ixaite_python_op
        from vllm_iluvatar.ilu_util.arch import cuda_arch

        arch = cuda_arch()
        if arch is None:
            raise RuntimeError("Iluvatar GPU architecture cannot be detected")
        ixaite.initialize(arch)
        fly, reason = load_ixaite_python_op("embedding_gather", "embedding_gather")
        if fly is None:
            raise RuntimeError(reason)
        return Providers(
            baseline_name="pytorch.F.embedding",
            baseline=F.embedding,
            flydsl=fly,
            flydsl_name="ixaite.embedding_gather",
        )

    def iter_cases(
        self, model: str, case_filter: str | None
    ) -> Sequence[EmbeddingGatherCase]:
        spec = l2_spec(model, self.name)
        vocab_size = int(spec["vocab_size"])
        hidden_size = int(spec["hidden_size"])
        cases = [
            EmbeddingGatherCase(
                case_id=f"embedding_gather@{model};T={num_tokens}",
                num_tokens=num_tokens,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
            )
            for num_tokens in (int(value) for value in spec["num_tokens"])
        ]
        return filter_cases(cases, case_filter)

    def gate_min_speedup(self) -> float:
        return 1.0


register(EmbeddingGatherSuite())

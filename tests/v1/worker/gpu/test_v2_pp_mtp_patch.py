# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the V2 PP + MTP plugin patch."""

from __future__ import annotations

import sys
from collections import deque
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.ilu_patch.general_patches import patch_v2_pp_mtp
from vllm_iluvatar.v1.worker.gpu import model_runner as runner_impl
from vllm_iluvatar.v1.worker.gpu import pp_utils as pp_impl

_RECEIVE_BASELINE_SHA = (
    "387a8574d05253e36757fa9bb5910bc27e38032b926fc57d317517b0bacb449e"
)


def test_v2_pp_mtp_receive_baseline_matches_declared_main_source():
    assert p.source_sha(patch_v2_pp_mtp.RECEIVE_BASELINE) == _RECEIVE_BASELINE_SHA


class _Tensor:
    """Tiny tensor stand-in for broadcast shape tests.

    The PPHandler wrappers only need shape/dtype/new_full/contiguous/record_stream
    for these unit tests, so this avoids requiring CUDA or NCCL.
    """

    def __init__(self, shape: tuple[int, ...], dtype: object = None) -> None:
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def new_full(self, shape: tuple[int, ...], fill_value: int):
        return _Tensor(shape, self.dtype)

    def to(self, dtype: object):
        return _Tensor(self.shape, dtype)

    def contiguous(self):
        return self

    def record_stream(self, stream: object) -> None:
        pass

    def unbind(self, dim: int = 0):
        assert dim == 0
        return [_Tensor(self.shape[1:], self.dtype) for _ in range(self.shape[0])]

    def __setitem__(self, key: object, value: object) -> None:
        pass


class _Stream:
    def wait_stream(self, stream: object) -> None:
        pass

    def record_event(self):
        return object()

    def wait_event(self, event: object) -> None:
        pass


class _StreamContext:
    def __init__(self, stream: _Stream) -> None:
        self.stream = stream

    def __enter__(self) -> _Stream:
        return self.stream

    def __exit__(self, *args: object) -> None:
        pass


class _PendingRecv:
    def __init__(
        self,
        event: object,
        sampled_tokens: _Tensor,
        num_sampled: _Tensor,
        num_rejected: _Tensor,
        idx_mapping: _Tensor,
        idx_mapping_np: np.ndarray,
        need_sampled_mask: np.ndarray,
        gen_at_receive_np: np.ndarray,
    ) -> None:
        self.event = event
        self.sampled_tokens = sampled_tokens
        self.num_sampled = num_sampled
        self.num_rejected = num_rejected
        self.idx_mapping = idx_mapping
        self.idx_mapping_np = idx_mapping_np
        self.need_sampled_mask = need_sampled_mask
        self.gen_at_receive_np = gen_at_receive_np


def test_v2_pp_mtp_enabled_reads_env_gate(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_envs = ModuleType("vllm.envs")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", fake_envs)

    fake_envs.VLLM_ILU_ENABLE_V2_PP_MTP = True
    assert patch_v2_pp_mtp._v2_pp_mtp_enabled() is True

    fake_envs.VLLM_ILU_ENABLE_V2_PP_MTP = False
    assert patch_v2_pp_mtp._v2_pp_mtp_enabled() is False


def test_pphandler_init_enables_opt_in_draft_relay_flag():
    def original_init(self, max_num_reqs, num_speculative_steps, device):
        self.args = (max_num_reqs, num_speculative_steps, device)

    wrapped = pp_impl.make_pphandler_init(original_init)
    handler = SimpleNamespace()

    wrapped(handler, 4, 3, "cpu", relay_draft_tokens=True)

    assert handler.args == (4, 3, "cpu")
    assert handler.relay_draft_tokens is True


def test_pphandler_get_prev_preserves_community_filtering_and_adds_drafts():
    slot = SimpleNamespace(draft_tokens="draft")

    def original_get_prev_sampled_outputs(self):
        self.queue.popleft()
        self.queue.append(None)
        return {"sampled_tokens": "sampled"}

    handler = SimpleNamespace(queue=deque([slot]))
    outputs = pp_impl.make_get_prev_sampled_outputs(original_get_prev_sampled_outputs)(
        handler
    )

    assert outputs == {"sampled_tokens": "sampled", "draft_tokens": "draft"}


def test_pphandler_broadcast_pads_sampled_tokens_to_fixed_width():
    broadcasts: list[tuple[int, ...]] = []

    def original_broadcast(
        self, sampled_token_ids, num_sampled, num_rejected, input_batch
    ):
        broadcasts.append(sampled_token_ids.shape)
        broadcasts.append((2, num_sampled.shape[0]))

    handler = SimpleNamespace(
        is_last_rank=True,
        max_sample_len=4,
    )
    input_batch = SimpleNamespace(num_reqs=2)
    sampled_token_ids = _Tensor((2, 1), pp_impl.torch.int64)
    num_sampled = _Tensor((2,))
    num_rejected = _Tensor((2,))

    pp_impl.make_broadcast(original_broadcast)(
        handler, sampled_token_ids, num_sampled, num_rejected, input_batch
    )

    assert broadcasts == [(2, 4), (2, 2)]


def test_pphandler_broadcast_draft_keeps_pr_compatible_natural_shape(monkeypatch):
    broadcasts: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        pp_impl.torch.cuda,
        "stream",
        lambda stream: _StreamContext(stream),
    )
    monkeypatch.setattr(
        pp_impl.torch.distributed,
        "broadcast",
        lambda tensor, **kwargs: broadcasts.append(tensor.shape),
    )

    pp_mod = SimpleNamespace(
        compute_need_sampled_mask=lambda input_batch: np.array([True, True])
    )
    handler = SimpleNamespace(
        is_last_rank=True,
        relay_draft_tokens=True,
        max_sample_len=4,
        last_rank=3,
        broadcast_group=object(),
        main_stream=_Stream(),
        broadcast_stream=_Stream(),
    )
    input_batch = SimpleNamespace(num_reqs=2)

    pp_impl.make_broadcast_draft(pp_mod)(
        handler, _Tensor((2, 2), pp_impl.torch.int64), input_batch
    )

    assert broadcasts == [(2, 2)]


def test_pphandler_receive_posts_matching_third_draft_collective(monkeypatch):
    broadcasts: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        pp_impl.torch.cuda,
        "stream",
        lambda stream: _StreamContext(stream),
    )
    monkeypatch.setattr(
        pp_impl.torch,
        "empty",
        lambda *shape, **kwargs: _Tensor(tuple(shape), kwargs.get("dtype")),
    )
    monkeypatch.setattr(
        pp_impl.torch.distributed,
        "broadcast",
        lambda tensor, **kwargs: broadcasts.append(tensor.shape),
    )

    pp_mod = SimpleNamespace(
        compute_need_sampled_mask=lambda input_batch: np.array([True, True]),
        PendingRecv=_PendingRecv,
    )

    handler = SimpleNamespace(
        is_last_rank=False,
        relay_draft_tokens=True,
        max_sample_len=4,
        last_rank=3,
        broadcast_group=object(),
        main_stream=_Stream(),
        broadcast_stream=_Stream(),
        device="cpu",
        req_idx_gen_np=np.array([0, 0], dtype=np.int32),
        queue=deque([None]),
    )
    input_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping=_Tensor((2,)),
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
    )

    assert pp_impl.make_receive(pp_mod)(handler, input_batch) is True

    assert broadcasts == [(2, 4), (2, 2), (2, 3)]
    assert handler.queue[-1].draft_tokens.shape == (2, 3)


def test_model_runner_update_applies_relayed_drafts_to_valid_rows():
    draft_state = np.full((3, 3), -1)

    def original_update_pp_decode_requests(self):
        self.postprocess_sampled(
            **{
                "sampled_tokens": "sampled",
                "num_sampled": "num_sampled",
                "num_rejected": "num_rejected",
                "idx_mapping": np.array([1, -1]),
                "draft_tokens": np.array([[10, 11, 12], [20, 21, 22]]),
            }
        )

    class Runner:
        req_states = SimpleNamespace(draft_tokens=draft_state)

        def __init__(self) -> None:
            self.postprocessed = None

        def postprocess_sampled(self, **outputs: Any) -> None:
            self.postprocessed = outputs

    runner = Runner()

    runner_impl.make_update_pp_decode_requests(original_update_pp_decode_requests)(
        runner
    )

    assert runner.postprocessed is not None
    assert runner.postprocessed["sampled_tokens"] == "sampled"
    assert runner.postprocessed["num_sampled"] == "num_sampled"
    assert runner.postprocessed["num_rejected"] == "num_rejected"
    np.testing.assert_array_equal(
        runner.postprocessed["idx_mapping"], np.array([1, -1])
    )
    np.testing.assert_array_equal(draft_state[1], np.array([10, 11, 12]))
    np.testing.assert_array_equal(draft_state[0], np.array([-1, -1, -1]))


def test_model_runner_wraps_propose_and_broadcasts_drafts_once():
    calls: list[tuple[object, object]] = []

    class PPHandler:
        def broadcast_draft(self, draft_tokens, input_batch):
            calls.append((draft_tokens, input_batch))

    class Speculator:
        supports_mm_inputs = False

        def propose(self, input_batch, *args, **kwargs):
            return "draft"

    runner = SimpleNamespace(pp_handler=PPHandler(), speculator=Speculator())
    input_batch = object()

    runner_impl._wrap_speculator_propose(runner)
    runner.speculator.propose(input_batch)
    runner_impl._wrap_speculator_propose(runner)
    runner.speculator.propose(input_batch)

    assert calls == [("draft", input_batch), ("draft", input_batch)]

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PP + MTP plugin-local compatibility helpers."""

from __future__ import annotations

import inspect
import sys
from types import ModuleType

from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.ilu_patch.general_patches import patch_pp_mtp
from vllm_iluvatar.v1.core.sched import scheduler as scheduler_impl
from vllm_iluvatar.v1.engine import core as core_impl
from vllm_iluvatar.v1.outputs import install_draft_token_ids_field
from vllm_iluvatar.v1.worker import gpu_model_runner as runner_impl

_BASELINE_SHAS = {
    "schedule": "2e35d747a76d483f899fa924dd463f81412323c93fc53ca31b209699213ac0e2",
    "update_states": "cb8fa7b1dfef46fd6e5142078a8d29ccb7ca32a2e93f4d9ead7f4eecb1a9ac7c",
    "build_attention_metadata": (
        "41580949c5c29971486dce920d4e7de2089f7a8973476c4359e719504e26a4c2"
    ),
    "dummy_run": "d70dcbf27ec5c7f0637de5020998cf7c4d44ca55f8bf2c579bba05204de46e95",
}


def test_pp_mtp_copied_method_baselines_match_declared_main_source():
    assert p.source_sha(scheduler_impl.SCHEDULE_BASELINE) == _BASELINE_SHAS["schedule"]
    assert (
        p.source_sha(runner_impl._UPDATE_STATES_BASELINE)
        == _BASELINE_SHAS["update_states"]
    )
    assert (
        p.source_sha(runner_impl._BUILD_ATTENTION_METADATA_BASELINE)
        == _BASELINE_SHAS["build_attention_metadata"]
    )
    assert p.source_sha(runner_impl._DUMMY_RUN_BASELINE) == _BASELINE_SHAS["dummy_run"]


def test_schedule_copy_keeps_negative_pp_token_guard():
    assert "if num_new_tokens <= 0:" in inspect.getsource(scheduler_impl._ilu_schedule)


def test_runner_copies_keep_pp_rank_and_intermediate_tensor_guards():
    update_states_source = inspect.getsource(runner_impl._ilu__update_states)
    metadata_source = inspect.getsource(runner_impl._ilu__build_attention_metadata)
    dummy_run_source = inspect.getsource(runner_impl._ilu__dummy_run)
    assert "not is_last_rank" in update_states_source
    assert "get_pp_group().is_last_rank" in metadata_source
    assert "isinstance(hidden_states, IntermediateTensors)" in dummy_run_source
    assert "seq_lens = max_query_len" in dummy_run_source
    assert "get_pp_group().is_last_rank" in dummy_run_source


def test_model_runner_output_accepts_and_retains_inline_drafts():
    class Output:
        def __init__(self, request_id: str) -> None:
            self.request_id = request_id

    install_draft_token_ids_field(Output)
    output = Output("r0", draft_token_ids=[[1, 2]])

    assert output.request_id == "r0"
    assert output.draft_token_ids == [[1, 2]]


def test_engine_post_step_skips_stale_rpc_for_pp_batch_queue():
    calls: list[bool] = []

    def original_post_step(_, model_executed: bool) -> None:
        calls.append(model_executed)

    wrapped = core_impl.make_post_step(original_post_step)

    class Core:
        batch_queue = object()
        use_spec_decode = True

    wrapped(Core(), True)

    assert calls == []


def test_sample_tokens_attaches_fresh_drafts_to_pp_output():
    class Output:
        req_ids = ["r0"]

    class Runner:
        draft_in_model_output = True

        def take_draft_token_ids(self):
            return [["draft"]]

    wrapped = runner_impl.make_sample_tokens(lambda *_args, **_kwargs: Output())
    output = wrapped(Runner(), None)

    assert output.draft_token_ids == [["draft"]]


def test_pp_mtp_enabled_reads_env_gate(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_envs = ModuleType("vllm.envs")
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.envs", fake_envs)

    fake_envs.VLLM_ILU_ENABLE_PP_MTP = True
    assert patch_pp_mtp._pp_mtp_enabled() is True

    fake_envs.VLLM_ILU_ENABLE_PP_MTP = False
    assert patch_pp_mtp._pp_mtp_enabled() is False


def test_pp_mtp_apply_skips_wiring_when_gate_disabled(monkeypatch):
    wired: list[bool] = []
    monkeypatch.setattr(patch_pp_mtp, "_pp_mtp_enabled", lambda: False)
    monkeypatch.setattr(patch_pp_mtp, "_wire_patches", lambda: wired.append(True))

    patch_pp_mtp._apply()

    assert wired == []


def test_pp_mtp_apply_wires_when_gate_enabled(monkeypatch):
    wired: list[bool] = []
    monkeypatch.setattr(patch_pp_mtp, "_pp_mtp_enabled", lambda: True)
    monkeypatch.setattr(patch_pp_mtp, "_wire_patches", lambda: wired.append(True))

    patch_pp_mtp._apply()

    assert wired == [True]


def _install_fake_pp_group(monkeypatch, is_last_rank):
    mod = ModuleType("vllm.distributed.parallel_state")

    class _PPGroup:
        pass

    _PPGroup.is_last_rank = is_last_rank
    mod.get_pp_group = lambda: _PPGroup()
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", mod)


def test_check_and_update_cudagraph_mode_hides_spec_off_last_rank(monkeypatch):
    _install_fake_pp_group(monkeypatch, is_last_rank=False)
    seen = {}

    class Runner:
        speculative_config = "SPEC"

    def original(self):
        seen["spec"] = self.speculative_config
        return "ret"

    wrapped = runner_impl.make_check_and_update_cudagraph_mode(original)
    runner = Runner()

    assert wrapped(runner) == "ret"
    assert seen["spec"] is None
    assert runner.speculative_config == "SPEC"


def test_check_and_update_cudagraph_mode_keeps_spec_on_last_rank(monkeypatch):
    _install_fake_pp_group(monkeypatch, is_last_rank=True)
    seen = {}

    class Runner:
        speculative_config = "SPEC"

    def original(self):
        seen["spec"] = self.speculative_config
        return "ret"

    wrapped = runner_impl.make_check_and_update_cudagraph_mode(original)
    runner = Runner()

    assert wrapped(runner) == "ret"
    assert seen["spec"] == "SPEC"


def test_initialize_metadata_builders_hides_spec_off_last_rank(monkeypatch):
    _install_fake_pp_group(monkeypatch, is_last_rank=False)
    seen = {}

    class Runner:
        speculative_config = "SPEC"

    def original(self):
        seen["spec"] = self.speculative_config
        return "ret"

    wrapped = runner_impl.make_initialize_metadata_builders(original)
    runner = Runner()

    assert wrapped(runner) == "ret"
    assert seen["spec"] is None
    assert runner.speculative_config == "SPEC"


def test_initialize_metadata_builders_keeps_spec_on_last_rank(monkeypatch):
    _install_fake_pp_group(monkeypatch, is_last_rank=True)
    seen = {}

    class Runner:
        speculative_config = "SPEC"

    def original(self):
        seen["spec"] = self.speculative_config
        return "ret"

    wrapped = runner_impl.make_initialize_metadata_builders(original)
    runner = Runner()

    assert wrapped(runner) == "ret"
    assert seen["spec"] == "SPEC"

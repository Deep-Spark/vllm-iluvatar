# SPDX-License-Identifier: Apache-2.0
"""CPU prefix hits remain visible; remote prefill transfers are not cache hits."""

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.metrics.stats import PrefillStats
from vllm_iluvatar.ilu_patch.general_patches import patch_pd_transfer_cached_tokens


@pytest.fixture(scope="module", autouse=True)
def apply_patch():
    patch_pd_transfer_cached_tokens._apply()


def add_request(processor, request_id, transfer):
    processor.add_request(
        EngineCoreRequest(
            request_id=request_id,
            external_req_id=request_id,
            prompt_token_ids=[1] * 4864,
            mm_features=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            sampling_params=SamplingParams(
                detokenize=False,
                extra_args={"kv_transfer_params": transfer} if transfer else None,
            ),
            pooling_params=None,
        ),
        prompt=None,
    )


def output(request_id, local, external):
    stats = PrefillStats()
    stats.set(4864, local, external)
    return EngineCoreOutput(
        request_id=request_id,
        new_token_ids=[42],
        prefill_stats=stats,
        finish_reason=FinishReason.LENGTH,
    )


@pytest.mark.parametrize(
    "transfer,local,external,expected",
    [
        (None, 0, 4096, 4096),
        ({"do_remote_decode": True}, 0, 4096, 4096),
        ({"do_remote_prefill": False}, 1024, 3072, 4096),
        (None, 4096, 0, 4096),
        ({"do_remote_prefill": True}, 0, 4096, 0),
        ({"do_remote_prefill": True}, 1024, 3072, 1024),
        (
            {"do_remote_prefill": True, "remote_prefill_cached_tokens": 2048},
            0,
            4096,
            2048,
        ),
    ],
    ids=["cpu", "producer_cpu", "mixed_cache", "gpu", "remote_cold",
         "remote_local_hit", "remote_producer_usage"],
)
def test_cache_usage_distinguishes_cpu_reuse_from_remote_prefill(
    transfer, local, external, expected
):
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    add_request(processor, "request", transfer)
    event = output("request", local, external)
    result = processor.process_outputs([event]).request_outputs[0]
    assert result.num_cached_tokens == expected
    assert result.outputs[0].token_ids == [42]
    assert event.prefill_stats.num_external_cached_tokens == external
    assert event.prefill_stats.num_computed_tokens == 4864 - local - external
    assert not processor.has_unfinished_requests()


def test_mixed_batch_and_reused_request_id_keep_independent_cache_accounting():
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    add_request(processor, "remote", {"do_remote_prefill": True})
    add_request(processor, "cpu", {"do_remote_decode": True})
    results = processor.process_outputs(
        [output("remote", 0, 4096), output("cpu", 0, 4096)]
    ).request_outputs
    assert {r.request_id: r.num_cached_tokens for r in results} == {
        "remote": 0,
        "cpu": 4096,
    }
    add_request(processor, "remote", None)
    result = processor.process_outputs([output("remote", 0, 4096)])
    assert result.request_outputs[0].num_cached_tokens == 4096

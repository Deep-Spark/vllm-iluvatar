from types import SimpleNamespace

import pytest

from vllm_iluvatar.models.deepseek_v41.model import _is_pure_prefill


@pytest.mark.parametrize(
    ("metadata", "num_tokens", "expected"),
    [
        (SimpleNamespace(num_prefill_tokens=256, num_decode_tokens=0), 256, True),
        (SimpleNamespace(num_prefill_tokens=0, num_decode_tokens=256), 256, False),
        (SimpleNamespace(num_prefill_tokens=128, num_decode_tokens=128), 256, False),
        (SimpleNamespace(num_prefill_tokens=256, num_decode_tokens=0), 128, False),
    ],
)
def test_pp_quant_requires_pure_prefill_batch(metadata, num_tokens, expected):
    assert _is_pure_prefill({"layer": metadata}, num_tokens) is expected


def test_pp_quant_rejects_profile_metadata():
    assert _is_pure_prefill(None, 256) is False
    assert _is_pure_prefill({}, 256) is False

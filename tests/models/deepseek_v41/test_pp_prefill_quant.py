from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from conftest import requires_corex30

from vllm_iluvatar.models.deepseek_v41 import _prefill_pp_codec as codec


@pytest.mark.parametrize(
    ("prefill", "decode", "expected"),
    [(256, 0, True), (0, 256, False), (192, 64, False), (255, 0, False)],
)
def test_pp_quant_requires_pure_prefill_batch(prefill, decode, expected):
    swa = MagicMock(spec=codec.DeepseekSparseSWAMetadata)
    swa.num_prefill_tokens = prefill
    swa.num_decode_tokens = decode
    # A different layer can report a different token count in the same batch.
    other = SimpleNamespace(num_prefill_tokens=256, num_decode_tokens=0)
    assert codec.is_pure_prefill({"other": other, "swa": swa}, 256) is expected
    assert not codec.is_pure_prefill({"other": other}, 256)


@pytest.mark.parametrize("metadata", [None, {}, []])
def test_pp_quant_rejects_profile_metadata(metadata):
    assert not codec.is_pure_prefill(metadata, 256)


@requires_corex30
@pytest.mark.parametrize(
    ("shape", "offset", "senders", "tp", "encoded"),
    [
        ((129, 512), 0, "0, 2", 1, True),
        ((129, 2, 512), 0, "all", 1, True),
        ((129, 511), 0, "all", 1, False),
        ((129, 512), 1, "all", 1, False),
        ((128, 512), 0, "all", 1, False),
        ((129, 512), 0, "1", 1, False),
        ((129, 512), 0, "all", 2, False),
    ],
)
def test_pp_sender_payload(monkeypatch, shape, offset, senders, tp, encoded):
    from math import prod

    swa = MagicMock(spec=codec.DeepseekSparseSWAMetadata)
    swa.num_prefill_tokens, swa.num_decode_tokens = shape[0], 0
    monkeypatch.setattr(codec, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(
        codec,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"layer": swa}),
    )
    group = SimpleNamespace(rank_in_group=2, use_cpu_custom_send_recv=False)
    x = torch.randn(prod(shape) + offset, dtype=torch.bfloat16, device="cuda")[
        offset:
    ].view(shape)
    payload = {"hidden_states": x}
    codec.maybe_encode_hidden_states(payload, senders, group)
    if not encoded:
        assert payload["hidden_states"] is x and "prefill_pp_scale" not in payload
        return
    q, scales = payload["hidden_states"], payload["prefill_pp_scale"]
    assert q.dtype == torch.int8 and q.shape == x.shape
    assert scales.shape == (shape[0], prod(shape[1:]) // 512)
    recovered = codec.decode(q, scales)
    error = (recovered.float() - x.float()).abs().view(*scales.shape, 512).amax(-1)
    # One quantization step includes BF16 rounding on reconstruction.
    assert (error <= scales).all()

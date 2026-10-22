"""CPU content identity and codec regression tests, including cache UUIDs."""

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch
from PIL import Image
from vllm.config import MultiModalConfig
from vllm.models.deepseek_v41.common.mm_preprocess import (
    DeepseekV4VLDummyInputsBuilder,
    DeepseekV4VLProcessingInfo,
)
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.multimodal.processing.context import InputProcessingContext, TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

from vllm_iluvatar.v1.worker.mm_replay.identity import (
    certified_image_id,
    feature_descriptors,
    model_identity,
    request_identity,
    token_digest,
)
from vllm_iluvatar.v1.worker.mm_replay.processor import CertifiedImageProcessor
from vllm_iluvatar.v1.worker.mm_replay.tensors import pack_tensors, unpack_tensors


def test_tensor_codec_exact_bf16_and_layout_validation():
    values = dict(
        types=torch.arange(8, dtype=torch.int64),
        patches=torch.arange(60, dtype=torch.float32)
        .to(torch.bfloat16)
        .reshape(5, 3, 2, 2),
        vit_grid=torch.tensor([1, 5]),
    )
    fields, raw = pack_tensors(values)
    restored = unpack_tensors(fields, raw)
    for key in values:
        assert torch.equal(
            values[key].view(torch.uint8), restored[key].view(torch.uint8)
        )
    broken = deepcopy(fields)
    broken[-1]["offset"] = 0
    with pytest.raises(ValueError):
        unpack_tensors(broken, raw)
    with pytest.raises(ValueError):
        unpack_tensors(fields, raw[:-1])


def test_certification_covers_content_grid_model_positions_and_order():
    fields, raw = pack_tensors(
        dict(patches=torch.zeros(4, 3, 2, 2, dtype=torch.bfloat16))
    )
    key = certified_image_id("model1", fields, raw)
    assert key != certified_image_id("model2", fields, raw)
    assert key != certified_image_id("model1", fields, raw[:-1] + b"1")
    features = [
        MultiModalFeatureSpec(
            None,
            "image",
            key,
            PlaceholderRange(4, 8, is_embed=torch.ones(8, dtype=torch.bool)),
        )
    ]
    desc = feature_descriptors(features)
    td = token_digest(list(range(129)))
    identity = request_identity("model1", td, 129, desc)
    assert identity != request_identity("model1", td, 129, [dict(desc[0], start=5)])
    features[0].identifier = "client-uuid"
    with pytest.raises(ValueError, match="certified"):
        feature_descriptors(features)


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return 129264

    def encode(self, text, **kwargs):
        return [129264] if text else []

    def decode(self, tokens, **kwargs):
        return "<｜deepseek_image｜>"


def test_real_processor_does_not_trust_reused_client_uuid():
    hf = DeepseekV41Config(vision_max_n_token=256, vision_min_pixels=1024)
    mm = MultiModalConfig(limit_per_prompt={"image": 5})
    mm._ilu_mm_replay = True
    model = NS(
        model="checkpoint-1017",
        hf_config=hf,
        dtype=torch.bfloat16,
        multimodal_config=mm,
        get_multimodal_config=lambda: mm,
        get_inputs_embeds_size=lambda: hf.hidden_size,
        max_model_len=1048576,
    )
    info = DeepseekV4VLProcessingInfo(InputProcessingContext(model, Tokenizer()))
    processor = CertifiedImageProcessor(info, DeepseekV4VLDummyInputsBuilder(info))
    results = []
    for color in ("red", "blue", "red"):
        items = info.parse_mm_data({"image": [Image.new("RGB", (64, 64), color)]})
        result = processor.apply(
            ProcessorInputs(
                [129264], items, mm_uuid_items={"image": ["same-untrusted-id"]}
            ),
            TimingContext(enabled=False),
        )
        results.append(result)
    assert results[0]["mm_hashes"]["image"] != results[1]["mm_hashes"]["image"]
    assert results[0]["mm_hashes"]["image"] == results[2]["mm_hashes"]["image"]
    item = results[0]["mm_kwargs"]["image"][0]
    fields, raw = pack_tensors(item.get_data())
    assert results[0]["mm_hashes"]["image"] == [
        certified_image_id(model_identity(model), fields, raw)
    ]
    assert all(x == 129264 for x in results[0]["prompt_token_ids"])


def test_real_replicated_lru_cache_one_then_five_images():
    from vllm.multimodal.cache.lru import (
        LruKeyReplicatedReceiverCache,
        LruKeyReplicatedSenderCache,
    )

    hf = DeepseekV41Config(vision_max_n_token=256, vision_min_pixels=1024)
    mm = MultiModalConfig(limit_per_prompt={"image": 5})
    mm._ilu_mm_replay = True
    model = NS(
        model="checkpoint-1017",
        hf_config=hf,
        dtype=torch.bfloat16,
        multimodal_config=mm,
        get_multimodal_config=lambda: mm,
        get_inputs_embeds_size=lambda: hf.hidden_size,
        max_model_len=1048576,
    )
    info = DeepseekV4VLProcessingInfo(InputProcessingContext(model, Tokenizer()))
    processor = CertifiedImageProcessor(info, DeepseekV4VLDummyInputsBuilder(info))
    sender = LruKeyReplicatedSenderCache(model)
    receiver = LruKeyReplicatedReceiverCache(model)
    colors = ["red", "blue", "green", "yellow", "purple"]
    previous = None
    for count in (1, 5, 5):
        items = info.parse_mm_data(
            {"image": [Image.new("RGB", (64, 64), c) for c in colors[:count]]}
        )
        result = processor.apply(
            ProcessorInputs([129264] * count, items, cache=sender),
            TimingContext(enabled=False),
        )
        hashes = result["mm_hashes"]["image"]
        kwargs = result["mm_kwargs"]["image"]
        if previous is not None:
            assert previous == hashes[0]
        for key, item in zip(hashes, kwargs, strict=True):
            restored = receiver.get_and_update_item(item, key)
            fields, raw = pack_tensors(restored.get_data())
            assert key == certified_image_id(model_identity(model), fields, raw)
        previous = hashes[0]

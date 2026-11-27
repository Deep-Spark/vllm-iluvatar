"""Certify image content before the processor/IPC caches select their keys."""

from dataclasses import dataclass, field, fields

from vllm.exceptions import VLLMValidationError
from vllm.models.deepseek_v41.common.mm_preprocess import (
    DeepseekV4VLMultiModalProcessor,
)
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.processing.inputs import ProcessorInputs

from .identity import certified_image_id, model_identity
from .tensors import pack_tensors


@dataclass
class _CertifiedInputs(ProcessorInputs):
    image_identities: list[str] = field(default_factory=list)

    def get_mm_hashes(self, model_id, hash_algorithm):
        hashes = super().get_mm_hashes(model_id, hash_algorithm)
        hashes["image"] = list(self.image_identities)
        return hashes


class CertifiedImageProcessor(DeepseekV4VLMultiModalProcessor):
    def apply(self, inputs, timing_ctx):
        mm_config = self.info.ctx.get_mm_config()
        if (
            not getattr(mm_config, "_ilu_mm_replay", False)
            or "image" not in inputs.mm_data_items
        ):
            return super().apply(inputs, timing_ctx)
        images = inputs.mm_data_items["image"]
        if any(images.get(i) is None for i in range(images.get_count())):
            raise VLLMValidationError(
                "P-only image replay requires image content; UUID-only inputs are unsupported",
                parameter="image",
            )
        # LRU sender cache hits deliberately carry no tensor data. Certify the
        # CPU preprocessing *before* cache lookup, so both the sender and its
        # replicated receiver use the same immutable content key. Never change
        # returned hashes after the cache has already committed its keys.
        with timing_ctx.record("certify_image_content"):
            processed = self._apply_hf_processor_main(
                mm_items=inputs.mm_data_items,
                hf_kwargs=inputs.hf_processor_mm_kwargs,
            )
            items = MultiModalKwargsItems.from_hf_inputs(
                processed,
                self._get_mm_fields_config(processed, inputs.hf_processor_mm_kwargs),
            )["image"]
            model_digest = model_identity(self.info.ctx.model_config)
            identities = []
            for item in items:
                layout, raw = pack_tensors(item.get_data())
                identities.append(certified_image_id(model_digest, layout, raw))
        values = {f.name: getattr(inputs, f.name) for f in fields(ProcessorInputs)}
        values["mm_uuid_items"] = None
        certified = _CertifiedInputs(**values, image_identities=identities)
        return super().apply(certified, timing_ctx)

"""Certified image identities, separate from the legacy token-only E25 digest."""

from __future__ import annotations

import hashlib
import json

CERTIFIED_PREFIX = "ilu-dsv41-image-v1:"


def canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode()


def model_identity(model_config) -> str:
    mm_config = getattr(model_config, "multimodal_config", None)
    frozen = getattr(mm_config, "_ilu_mm_model_identity", None)
    if frozen is not None:
        return frozen
    config = dict(model_config.hf_config.to_dict())
    config.pop("_name_or_path", None)
    return hashlib.sha256(
        canonical_json(
            dict(
                model=str(model_config.model),
                config=config,
                dtype=str(model_config.dtype),
            )
        )
    ).hexdigest()


def certified_image_id(model_digest: str, fields: list[dict], raw: bytes) -> str:
    digest = hashlib.sha256(
        canonical_json(dict(version=1, model=model_digest, fields=fields))
    )
    digest.update(raw)
    return CERTIFIED_PREFIX + digest.hexdigest()


def feature_descriptors(features) -> list[dict]:
    descriptors = []
    for f in features:
        key = f.identifier
        position = f.mm_position
        if (
            f.modality != "image"
            or not key.startswith(CERTIFIED_PREFIX)
            or len(key) != len(CERTIFIED_PREFIX) + 64
            or any(c not in "0123456789abcdef" for c in key[len(CERTIFIED_PREFIX) :])
        ):
            raise ValueError(
                "P-only encoding requires certified processed image content"
            )
        mask = position.is_embed
        if mask is not None and (
            mask.device.type != "cpu"
            or mask.numel() != position.length
            or not bool(mask.all())
        ):
            raise ValueError(
                "Sparse image span masks are not supported by DeepSeek V4.1 replay"
            )
        start, rows = position.offset, position.length
        if type(start) is not int or type(rows) is not int or start < 0 or rows <= 0:
            raise ValueError("Invalid absolute image range")
        if descriptors and start < descriptors[-1]["start"] + descriptors[-1]["rows"]:
            raise ValueError("Image spans must be ordered and disjoint")
        descriptors.append(dict(key=key, start=start, rows=rows))
    return descriptors


def request_identity(model_digest, token_digest, prompt_len, descriptors) -> str:
    return hashlib.sha256(
        canonical_json(
            dict(
                version=1,
                model=model_digest,
                token_sha256=token_digest,
                prompt_len=prompt_len,
                features=descriptors,
            )
        )
    ).hexdigest()


def token_digest(tokens) -> str:
    import numpy as np

    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def overlapping(features, start, end):
    return [
        dict(
            f,
            start=max(start, f["start"]),
            rows=min(end, f["start"] + f["rows"]) - max(start, f["start"]),
        )
        for f in features
        if max(start, f["start"]) < min(end, f["start"] + f["rows"])
    ]

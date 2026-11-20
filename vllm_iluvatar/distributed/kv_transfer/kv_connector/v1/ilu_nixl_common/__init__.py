# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared geometry, semantic pairing, RLE codecs, and FP8 staging primitives."""

from .memory_registration import (
    _U32_MAX,
    _U64_MAX,
    MemoryRegistrationPlanner,
    PackedLayerInfo,
    TensorRegistration,
    validate_64bit_span,
    validate_registered_allocation_alias,
)
from .pool_directory import PoolDirectory
from .types import (
    HmaTransferRoute,
    RegionPairing,
    SemanticTransferRegion,
    decode_block_ids_rle,
    encode_block_ids_rle,
    resolve_remote_block_ids,
)

__all__ = [
    "_U32_MAX",
    "_U64_MAX",
    "HmaTransferRoute",
    "MemoryRegistrationPlanner",
    "PackedLayerInfo",
    "PoolDirectory",
    "RegionPairing",
    "SemanticTransferRegion",
    "TensorRegistration",
    "decode_block_ids_rle",
    "encode_block_ids_rle",
    "resolve_remote_block_ids",
    "validate_64bit_span",
    "validate_registered_allocation_alias",
]

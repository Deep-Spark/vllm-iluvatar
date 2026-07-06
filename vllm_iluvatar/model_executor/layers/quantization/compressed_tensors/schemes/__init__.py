# SPDX-License-Identifier: Apache-2.0
"""Iluvatar compressed-tensors linear schemes."""

from .ilu_compressed_tensors_w4a8_int import IluCompressedTensorsW4A8Int
from .ilu_compressed_tensors_w8a8_int8 import IluCompressedTensorsW8A8Int8

__all__ = [
    "IluCompressedTensorsW4A8Int",
    "IluCompressedTensorsW8A8Int8",
]

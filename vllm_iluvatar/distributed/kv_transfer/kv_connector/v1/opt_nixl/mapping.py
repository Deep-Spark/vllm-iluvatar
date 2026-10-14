"""Unified mapping exports for NIXL handshake planning.

This module is the production import surface for PP/TP/layer mapping helpers.
The implementation remains split by semantic layer to keep the large TP layout
logic isolated and low-risk.
"""

from .layer_mapping import (
    Descriptor,
    LayerDescriptorMapping,
    LayerMappingPlanner,
    LayerSelection,
    PointerPair,
    RegistrationView,
)
from .pipeline_mapping import PipelineMappingPlanner, PipelineSlice
from .tp_layout import AdjustedXferLayout, AdjustedXferSide, TpLayoutAdjuster
from .tp_mapping import NixlTpKVTopology, TensorParallelMapping

__all__ = [
    "AdjustedXferLayout",
    "AdjustedXferSide",
    "Descriptor",
    "LayerDescriptorMapping",
    "LayerMappingPlanner",
    "LayerSelection",
    "NixlTpKVTopology",
    "PipelineMappingPlanner",
    "PipelineSlice",
    "PointerPair",
    "RegistrationView",
    "TensorParallelMapping",
    "TpLayoutAdjuster",
]

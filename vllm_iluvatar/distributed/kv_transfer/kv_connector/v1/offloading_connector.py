# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py

Target   : OffloadingConnector.set_xfer_handshake_metadata_pp_aware
Baseline : v0.31.0 (341bf7d4d124)
Why      : Native local CPU offload has no remote PP peer metadata.
How      : Ignore sibling OptNixl metadata in MultiConnector handshake fanout.
Scope    : Local offload connector in MultiConnector PP handshake fanout.
Remove   : When community local offload handles MultiConnector PP handshake fanout.
"""

from __future__ import annotations


# [ILU override] _ilu_offloading_set_xfer_handshake_metadata_pp_aware — delta: adapt paired pages / rank-local PP geometry.
def _ilu_offloading_set_xfer_handshake_metadata_pp_aware(
    self,
    metadata,
) -> None:
    """Ignore a sibling OptNixl connector's remote PP handshake metadata."""

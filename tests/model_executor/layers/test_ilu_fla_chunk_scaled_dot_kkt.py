# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the Iluvatar FLA chunk_scaled_dot_kkt patch."""

from __future__ import annotations

import vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt as community

from vllm_iluvatar.ilu_patch.general_patches import patch_chunk_scaled_dot_kkt
from vllm_iluvatar.ilu_patch import general_patch as p
from vllm_iluvatar.model_executor.layers.fla.ops import (
    chunk_scaled_dot_kkt as ilu_impl,
)


def test_chunk_scaled_dot_kkt_patch_rebinds_from_import_consumer(monkeypatch):
    community_wrapper = community.chunk_scaled_dot_kkt_fwd

    # Some local test environments import a legacy forked vLLM wheel where this
    # kernel already carries the Iluvatar line. Keep this test focused on the
    # rebind mechanism while leaving production fingerprint checking intact.
    monkeypatch.setattr(
        patch_chunk_scaled_dot_kkt,
        "_kernel_source",
        lambda _kernel: patch_chunk_scaled_dot_kkt.CHUNK_SCALED_DOT_KKT_FWD_KERNEL,
    )
    patch_chunk_scaled_dot_kkt._apply()

    assert community.chunk_scaled_dot_kkt_fwd is community_wrapper
    assert community.chunk_scaled_dot_kkt_fwd_kernel is (
        ilu_impl.chunk_scaled_dot_kkt_fwd_kernel
    )
    assert community.chunk_scaled_dot_kkt_fwd.__globals__[
        "chunk_scaled_dot_kkt_fwd_kernel"
    ] is ilu_impl.chunk_scaled_dot_kkt_fwd_kernel


def test_chunk_scaled_dot_kkt_kernel_signature_matches_community():
    patch_chunk_scaled_dot_kkt._check_kernel_signature(
        community.chunk_scaled_dot_kkt_fwd_kernel,
        ilu_impl.chunk_scaled_dot_kkt_fwd_kernel,
    )


def test_chunk_scaled_dot_kkt_kernel_baseline_and_ilu_delta():
    baseline_source = p.normalize_src(
        patch_chunk_scaled_dot_kkt.CHUNK_SCALED_DOT_KKT_FWD_KERNEL
    )
    ilu_source = p.normalize_src(
        patch_chunk_scaled_dot_kkt._kernel_source(
            ilu_impl.chunk_scaled_dot_kkt_fwd_kernel
        )
    )

    assert "tl.dot(b_kb, tl.trans(b_k).to(b_kb.dtype))" in baseline_source
    assert "tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))" in ilu_source

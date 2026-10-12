import pytest
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

from vllm_iluvatar.ilu_patch.general_patches import patch_ilu_indexer


def test_baseline_matches_installed_community_build():
    patch_ilu_indexer._check_community_build(DeepseekV32IndexerMetadataBuilder.build)


def test_community_build_drift_fails(monkeypatch):
    monkeypatch.setenv("VLLM_ILU_PATCH_CHECK_FINGERPRINT", "1")
    monkeypatch.setattr(patch_ilu_indexer, "_COMMUNITY_BUILD_SOURCE_SHA", "0" * 64)

    with pytest.raises(RuntimeError, match="community .* source changed"):
        patch_ilu_indexer._check_community_build(
            DeepseekV32IndexerMetadataBuilder.build
        )


def test_fingerprint_switch_skips_check(monkeypatch):
    monkeypatch.setattr(patch_ilu_indexer.p, "_enabled", lambda env: False)
    monkeypatch.setattr(patch_ilu_indexer, "_COMMUNITY_BUILD_SOURCE_SHA", "0" * 64)

    patch_ilu_indexer._check_community_build(DeepseekV32IndexerMetadataBuilder.build)

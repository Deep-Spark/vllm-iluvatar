import torch

from vllm_iluvatar.ilu_patch.general_patches.patch_mamba_hybrid_state import (
    make_postprocess_state,
)


def test_python_int_num_sampled_converts_idx_mapping_to_int64():
    captured = {}

    def original(self, idx_mapping, num_sampled, num_computed_tokens=None):
        captured["idx_mapping"] = idx_mapping

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    make_postprocess_state(original)(None, idx_mapping, 0)

    assert captured["idx_mapping"].dtype == torch.int64
    assert idx_mapping.dtype == torch.int32


def test_tensor_num_sampled_keeps_idx_mapping_dtype():
    captured = {}

    def original(self, idx_mapping, num_sampled, num_computed_tokens=None):
        captured["idx_mapping"] = idx_mapping

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_sampled = torch.tensor([1, 1], dtype=torch.int32)
    make_postprocess_state(original)(None, idx_mapping, num_sampled)

    assert captured["idx_mapping"] is idx_mapping

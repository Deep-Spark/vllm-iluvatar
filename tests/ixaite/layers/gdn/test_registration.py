# SPDX-License-Identifier: Apache-2.0
"""Verify private GDN selection without changing baseline discovery tests."""


# The private mapping must retain the GDN replacement after moving its tests.
def test_qwen_gdn_oot_module_is_selected_for_ixaite():
    from vllm_iluvatar.backends.ixaite.oot_registration import _SELECTED_OP_MODULES

    assert "vllm_iluvatar.model_executor.layers.mamba.gdn.ilu_qwen_gdn_linear_attn" in _SELECTED_OP_MODULES

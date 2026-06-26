# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the fused_moe layer_patch (rebind MoERunner / SharedExperts).

Verifies that ``layer_patch._apply()`` correctly rebinds the community
``MoERunner`` and ``SharedExperts`` to their Iluvatar subclasses, including
propagation to ``from-import`` consumers (e.g. ``layer.py``).

No GPU required — the tests exercise the pure-Python ``rebind_module_func``
machinery, following the same pattern as ``test_fa_utils_patch.py``.
"""

from __future__ import annotations

import vllm.model_executor.layers.fused_moe.layer as layer_mod
import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner_mod
import vllm.model_executor.layers.fused_moe.runner.shared_experts as se_mod

from vllm_iluvatar.model_executor.layers.fused_moe import layer_patch
from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_moe_runner import (
    IluMoERunner,
)
from vllm_iluvatar.model_executor.layers.fused_moe.runner.ilu_shared_experts import (
    IluSharedExperts,
)


def test_layer_patch_rebinds_moe_runner():
    """After _apply(), MoERunner on both the definition module and the
    from-import consumer (layer.py) must be IluMoERunner."""
    layer_patch._apply()

    assert moe_runner_mod.MoERunner is IluMoERunner, (
        "MoERunner on runner module not rebound"
    )
    assert layer_mod.MoERunner is IluMoERunner, (
        "MoERunner on layer.py (from-import) not propagated"
    )


def test_layer_patch_rebinds_shared_experts():
    """After _apply(), SharedExperts on the definition module must be
    IluSharedExperts."""
    layer_patch._apply()

    assert se_mod.SharedExperts is IluSharedExperts, "SharedExperts not rebound"


def test_layer_patch_register_is_idempotent():
    """Calling register() multiple times must not raise and must apply
    only once (guarded by general_patch.register_once)."""
    layer_patch.register()
    layer_patch.register()
    layer_patch.register()

    assert moe_runner_mod.MoERunner is IluMoERunner
    assert se_mod.SharedExperts is IluSharedExperts


def test_ilu_moe_runner_is_subclass():
    """IluMoERunner must be a proper subclass of the community MoERunner.

    After rebind, ``from vllm...import MoERunner`` resolves to IluMoERunner
    itself, so we verify the inheritance via the MRO instead.
    """
    mro_names = [cls.__name__ for cls in IluMoERunner.__mro__]
    assert "MoERunner" in mro_names, f"MoERunner not in MRO: {mro_names}"
    assert mro_names.index("IluMoERunner") < mro_names.index("MoERunner"), (
        "IluMoERunner must appear before MoERunner in MRO"
    )


def test_ilu_shared_experts_is_subclass():
    """IluSharedExperts must be a proper subclass of the community
    SharedExperts. Verified via MRO (same rebind caveat as above)."""
    mro_names = [cls.__name__ for cls in IluSharedExperts.__mro__]
    assert "SharedExperts" in mro_names, f"SharedExperts not in MRO: {mro_names}"
    assert mro_names.index("IluSharedExperts") < mro_names.index("SharedExperts"), (
        "IluSharedExperts must appear before SharedExperts in MRO"
    )

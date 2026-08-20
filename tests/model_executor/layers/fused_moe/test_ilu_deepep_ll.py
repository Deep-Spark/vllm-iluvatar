# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from vllm.model_executor.layers.fused_moe.config import (
    int8_w8a8_moe_quant_config,
    mxfp4_w4a8_moe_quant_config,
)

from vllm_iluvatar.model_executor.layers.fused_moe.prepare_finalize import (
    ilu_deepep_ll,
)
from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe_ep import (
    IluFusedMoEEP,
)


def test_nvsm_dispatch_quant_kwargs_follow_moe_quant_config(monkeypatch):
    monkeypatch.setattr(
        ilu_deepep_ll.current_platform,
        "fp8_dtype",
        lambda: torch.float8_e4m3fn,
    )

    int8_config = int8_w8a8_moe_quant_config(
        w1_scale=torch.ones(1),
        w2_scale=torch.ones(1),
        a1_scale=None,
        a2_scale=None,
        per_act_token_quant=True,
    )
    fp4_fp8_config = mxfp4_w4a8_moe_quant_config(
        w1_scale=torch.ones(1),
        w2_scale=torch.ones(1),
    )

    assert ilu_deepep_ll.nvsm_dispatch_quant_kwargs(int8_config) == {
        "use_int8": True,
        "use_fp8": False,
    }
    assert ilu_deepep_ll.nvsm_dispatch_quant_kwargs(fp4_fp8_config) == {
        "use_int8": False,
        "use_fp8": True,
        "fp8_format": torch.float8_e4m3fn,
        "fp8_group_size": None,
    }

    ep = IluFusedMoEEP()
    ep.experts = SimpleNamespace(
        _quant_method=SimpleNamespace(moe_quant_config=fp4_fp8_config)
    )
    assert ep._nvsm_dispatch_quant_kwargs() == {
        "use_int8": False,
        "use_fp8": True,
        "fp8_format": torch.float8_e4m3fn,
        "fp8_group_size": None,
    }

    ep.experts._quant_method.moe_quant_config = int8_config
    assert ep._nvsm_dispatch_quant_kwargs() == {
        "use_int8": True,
        "use_fp8": False,
    }


def test_nvsm_dispatch_quant_kwargs_cover_quant_dtype_families(monkeypatch):
    fp8_dtype = torch.float8_e4m3fn
    monkeypatch.setattr(
        ilu_deepep_ll.current_platform,
        "fp8_dtype",
        lambda: fp8_dtype,
    )
    cases = (
        (None, None, False, False),
        (None, "int4", True, False),
        (None, torch.int8, True, False),
        ("mxint8", "int4", True, False),
        (fp8_dtype, fp8_dtype, False, True),
        ("fp4", "fp4", False, True),
        ("fp8", "mxfp4", False, True),
        ("mxfp4", "mxfp4", False, True),
        ("mxfp8", "mxfp4", False, True),
        ("mxfp8_e4m3", "mxfp4", False, True),
        ("mxfp8_e5m2", "mxfp4", False, True),
        ("nvfp4", "nvfp4", False, True),
        (fp8_dtype, "int4", False, True),
    )

    for quant_dtype, weight_dtype, use_int8, use_fp8 in cases:
        config = SimpleNamespace(
            quant_dtype=quant_dtype,
            weight_quant_dtype=weight_dtype,
        )
        kwargs = ilu_deepep_ll.nvsm_dispatch_quant_kwargs(config)

        assert kwargs["use_int8"] is use_int8
        assert kwargs["use_fp8"] is use_fp8
        if use_fp8:
            assert kwargs["fp8_format"] == fp8_dtype
            assert kwargs["fp8_group_size"] is None
        else:
            assert "fp8_format" not in kwargs
            assert "fp8_group_size" not in kwargs

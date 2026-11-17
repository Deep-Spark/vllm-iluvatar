# SPDX-License-Identifier: Apache-2.0
"""CPU call-path tests with fully isolated binary and Tensor stand-ins.

No Torch/vLLM/ixformer imports or device access. Native numerical and graph
contracts are tested separately on the paired CQ runtime.
"""

import __future__

import ast
import math
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
NN = "vllm_iluvatar/model_executor/layers/quantization/backends/ivcore30/w4a8/nn_moe.py"
STD = (
    "vllm_iluvatar/model_executor/layers/quantization/"
    "backends/ivcore30/w4a8/nn_experts.py"
)
ROUTER = (
    "vllm_iluvatar/model_executor/layers/fused_moe/router/ilu_grouped_topk_router.py"
)
OPT = "vllm_iluvatar/envs.py"


def load(path, name, namespace, cls=None):
    tree = ast.parse((ROOT / path).read_text())
    if cls:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            path,
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


class Tensor:
    def __init__(
        self, shape=(3, 256), dtype="bf16", device="cuda", contiguous=True, value=1
    ):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.contiguous_flag, self.value = contiguous, value

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_cuda(self):
        return self.device == "cuda"

    def is_contiguous(self):
        return self.contiguous_flag

    def numel(self):
        return math.prod(self.shape)

    def reshape(self, *shape):
        if -1 in shape:
            n = self.numel() // math.prod(x for x in shape if x != -1)
            shape = tuple(n if x == -1 else x for x in shape)
        return Tensor(shape, self.dtype, self.device)

    def contiguous(self):
        return self

    def to(self, dtype=None, **kwargs):
        return Tensor(self.shape, dtype or self.dtype, self.device, value=self.value)

    def __mul__(self, scale):
        return Tensor(self.shape, self.dtype, self.device, value=self.value * scale)


TORCH = SimpleNamespace(
    int8="int8",
    int32="int32",
    bfloat16="bf16",
    float16="fp16",
    float32="fp32",
    empty=lambda shape, dtype="fp32", device="cuda": Tensor(shape, dtype, device),
)


def modules(enabled=True, arch="ivcore30", ops=None):
    names = [
        "vllm",
        "vllm.logger",
        "vllm_iluvatar",
        "vllm.envs",
        "vllm_iluvatar.ilu_util",
        "vllm_iluvatar.ilu_util.arch",
        "vllm_iluvatar.model_executor",
        "vllm_iluvatar.model_executor.layers",
        "vllm_iluvatar.model_executor.layers.quantization",
        "vllm_iluvatar.model_executor.layers.quantization.capabilities",
        "ixformer",
        "ixformer.inference",
        "ixformer.inference.functions",
        "ixformer.inference.backend_registry",
    ]
    result = {name: ModuleType(name) for name in names}
    for module in result.values():
        module.__path__ = []
    opt = result["vllm.envs"]
    opt.VLLM_ILU_GLM_MOE_PREPARE_FUSION = enabled
    is_ivcore30 = lambda: arch == "ivcore30"  # noqa: E731
    result["vllm_iluvatar.ilu_util.arch"].is_ivcore30 = is_ivcore30
    # Quantization code reads the predicate through `capabilities`, which binds
    # its own name at import time, so stubbing `ilu_util.arch` alone is not seen.
    result[
        "vllm_iluvatar.model_executor.layers.quantization.capabilities"
    ].is_ivcore30 = is_ivcore30
    result["vllm.logger"].init_logger = lambda _: SimpleNamespace(
        info_once=lambda *a: None
    )
    result["ixformer.inference.backend_registry"].current_cuda_arch = lambda: arch
    if ops:
        for name in ("moe_grouped_topk", "moe_sigmoid_topk"):
            setattr(result["ixformer.inference.functions"], name, getattr(ops, name))
    return result


class Ops:
    def __init__(self, fail_native=False):
        self.calls = []
        self.fail_native = fail_native

    def scaled_int8_quant(self, x):
        self.calls.append("quant")
        return Tensor(x.shape, "int8", x.device), Tensor(
            (x.shape[0], 1), "fp32", x.device
        )

    def moe_expand_input_ep(self, **kwargs):
        self.calls.append("expand")

    def moe_expand_input_ep_dynamic_scaled_int8(self, *args, **kwargs):
        self.calls.append("fused")
        if self.fail_native:
            raise RuntimeError("native sentinel")

    def moe_compute_token_index_ep_packed(self, **kwargs):
        return (
            Tensor((24,), "int32"),
            None,
            Tensor((24,)),
            Tensor((4,), "int32"),
            None,
            Tensor((1,), "int32"),
        )

    def moe_w4a8_group_gemm_ixmm(self, **kwargs):
        return Tensor((24, 256))

    def moe_output_reduce_sum_ep(self, **kwargs):
        return kwargs.get("output", Tensor())

    def moe_grouped_topk(self, *args, **kwargs):
        self.calls.append("router_old")
        scale = kwargs.get("routed_scaling_factor", 1.0)
        return Tensor((3, 8), "fp32", value=scale), Tensor((3, 8), "int32")

    def moe_sigmoid_topk(self, *args, **kwargs):
        self.calls.append("router_native")
        if self.fail_native:
            raise RuntimeError("native sentinel")
        return Tensor((3, 8), "fp32"), Tensor((3, 8), "int32")


def nn(ops):
    ns = {
        "torch": TORCH,
        "ixfops": ops,
        "unpad_tensor": lambda x, *a, **k: x,
        "_get_ilu_output_scale": lambda layer: layer._ilu_output_scale,
    }
    cls = "IluW4A8NNMoEMethod"
    unpack = load(NN, "_unpack_input", ns, cls)
    method = load(NN, "_apply_group_gemm", ns, cls)
    self = SimpleNamespace(
        gemm1_info=SimpleNamespace(pad_k=256),
        gemm2_info=SimpleNamespace(get_n=lambda: 256),
        _group_gemm=lambda *a: Tensor((24, 256)),
        _activation_and_quant=lambda **k: (
            Tensor((24, 256), "int8"),
            Tensor((24,), "fp32"),
        ),
    )
    self._unpack_input = lambda x: unpack(self, x)
    layer = SimpleNamespace(
        expert_map=None,
        local_num_experts=4,
        global_num_experts=8,
        top_k=8,
        w13_input_scale=None,
        w13_weight=Tensor(),
        w13_weight_scale=Tensor(),
        w2_weight=Tensor(),
        w2_weight_scale=Tensor(),
        _ilu_output_scale=0.375,
    )
    return self, layer, method


class MoECPU(unittest.TestCase):
    def test_flag_reader(self):
        for value, expected in (
            (None, False),
            ("0", False),
            ("1", True),
            ("true", True),
            (" YES ", True),
            ("on", True),
            ("2", False),
            ("", False),
        ):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["VLLM_ILU_GLM_MOE_PREPARE_FUSION"] = value
                ns = {}
                exec(compile((ROOT / OPT).read_text(), OPT, "exec"), ns)
                self.assertIs(
                    ns["_ILU_CANONICAL_ONLY_ENV_VARS"][
                        "VLLM_ILU_GLM_MOE_PREPARE_FUSION"
                    ](),
                    expected,
                )

    def test_unpack_valid_and_invalid(self):
        self_nn, _, _ = nn(Ops())
        for dtype in ("bf16", "fp16"):
            x = Tensor(dtype=dtype)
            self.assertEqual(self_nn._unpack_input(x), (x, None, dtype))
            q, scale = Tensor(dtype="int8"), Tensor((3,), "fp32")
            actual = self_nn._unpack_input((q, scale, dtype))
            self.assertIs(actual[0], q)
            self.assertEqual(actual[1].shape, (3, 1))
            self.assertEqual(actual[2], dtype)
        bad = [
            (Tensor(dtype="bf16"), Tensor((3,), "fp32"), "bf16"),
            (Tensor((3, 192), "int8"), Tensor((3,), "fp32"), "bf16"),
            (Tensor((768,), "int8"), Tensor((3,), "fp32"), "bf16"),
            (Tensor(dtype="int8"), Tensor((1,), "fp32"), "bf16"),
            (Tensor(dtype="int8"), Tensor((3,), "bf16"), "bf16"),
            (Tensor(dtype="int8"), Tensor((3,), "fp32", "cpu"), "bf16"),
            (Tensor(dtype="int8"), Tensor((3,), "fp32"), "fp32"),
        ]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self_nn._unpack_input(args)

    def test_apply_passes_tensor_to_residual_check(self):
        fn = load(NN, "apply", {}, "IluW4A8NNMoEMethod")
        q, scale = Tensor(dtype="int8"), Tensor((3,), "fp32")
        inputs = (q, scale, "bf16")
        seen = []
        obj = SimpleNamespace(
            _valid_extra_residual=lambda x, s: seen.append(x),
            _apply_group_gemm=lambda layer, x, *a: x,
        )
        self.assertIs(fn(obj, None, inputs, None, None, None, None), inputs)
        self.assertEqual(seen, [q])

    def test_monolithic_call_paths_and_prequantized_bypass(self):
        for enabled in (False, True):
            for arch in (None, "ivcore11", "ivcore30"):
                for x in (
                    Tensor(),
                    Tensor(dtype="fp16"),
                    Tensor(device="cpu"),
                    Tensor(contiguous=False),
                    Tensor((3, 192)),
                ):
                    with self.subTest(
                        enabled=enabled,
                        arch=arch,
                        shape=x.shape,
                        dtype=x.dtype,
                        device=x.device,
                    ):
                        ops = Ops()
                        obj, layer, fn = nn(ops)
                        with patch.dict(sys.modules, modules(enabled, arch)):
                            fn(
                                obj,
                                layer,
                                x,
                                Tensor((3, 8), "fp32"),
                                Tensor((3, 8), "int32"),
                                None,
                            )
                        fused = (
                            enabled
                            and arch == "ivcore30"
                            and x.dtype == "bf16"
                            and x.is_cuda
                            and x.is_contiguous()
                            and x.shape[1] == 256
                        )
                        self.assertEqual(
                            ops.calls, ["fused"] if fused else ["quant", "expand"]
                        )
            ops = Ops()
            obj, layer, fn = nn(ops)
            with patch.dict(sys.modules, modules(enabled)):
                fn(
                    obj,
                    layer,
                    (Tensor(dtype="int8"), Tensor((3,), "fp32"), "bf16"),
                    Tensor((3, 8)),
                    Tensor((3, 8), "int32"),
                    None,
                )
            self.assertEqual(ops.calls, ["expand"])

    def test_standard_call_paths(self):
        for enabled in (False, True):
            for arch in (None, "ivcore11", "ivcore30"):
                for x in (
                    Tensor(),
                    Tensor(dtype="fp16"),
                    Tensor(device="cpu"),
                    Tensor(contiguous=False),
                    Tensor((3, 192)),
                ):
                    ops = Ops()
                    ns = {
                        "torch": TORCH,
                        "ixff": ops,
                        "get_ep_group": lambda: SimpleNamespace(
                            rank=1, rank_in_group=0
                        ),
                        "unpad_tensor": lambda x, *a, **k: x,
                    }
                    fn = load(STD, "_apply_nn", ns, "IluMoEStandardExpertsW4A8NN")
                    obj = SimpleNamespace(
                        group_size=-1,
                        w13_input_scale=None,
                        local_num_experts=4,
                        gemm1_info=SimpleNamespace(pad_k=256),
                        gemm2_info=SimpleNamespace(get_n=lambda: 256),
                        w1_scale=Tensor((4, 512), "fp32"),
                        w2_scale=Tensor((4, 256), "fp32"),
                        runner_scaling_factor=0.375,
                        _activation_and_quant=lambda **k: (
                            Tensor((24, 256), "int8"),
                            Tensor((24,), "fp32"),
                        ),
                    )
                    with patch.dict(sys.modules, modules(enabled, arch)):
                        fn(
                            obj,
                            Tensor(),
                            x,
                            Tensor((4, 512, 256)),
                            Tensor((4, 256, 256)),
                            Tensor((3, 8)),
                            Tensor((3, 8), "int32"),
                            None,
                            8,
                        )
                    fused = (
                        enabled
                        and arch == "ivcore30"
                        and x.dtype == "bf16"
                        and x.is_cuda
                        and x.is_contiguous()
                        and x.shape[1] == 256
                    )
                    self.assertEqual(
                        ops.calls, ["fused"] if fused else ["quant", "expand"]
                    )

    def test_router_selection_and_scaling_fallback(self):
        for enabled, arch, topk, scoring in [
            (True, "ivcore30", 8, "sigmoid"),
            (False, "ivcore30", 8, "sigmoid"),
            (True, None, 8, "sigmoid"),
            (True, "ivcore30", 6, "sigmoid"),
            (True, "ivcore30", 8, "softmax"),
        ]:
            ops = Ops()
            fn = load(ROUTER, "grouped_topk", {"torch": TORCH})
            with patch.dict(sys.modules, modules(enabled, arch, ops)):
                weights, _ = fn(
                    Tensor(),
                    Tensor(),
                    topk,
                    True,
                    1,
                    1,
                    scoring,
                    0.375,
                    Tensor((256,), "fp32"),
                )
            fused = (
                enabled and arch == "ivcore30" and topk == 8 and scoring == "sigmoid"
            )
            self.assertEqual(ops.calls, ["router_native"] if fused else ["router_old"])
            if not fused:
                self.assertEqual(weights.value, 0.375)

    def test_native_errors_are_not_silent_fallback(self):
        ops = Ops(fail_native=True)
        obj, layer, fn = nn(ops)
        with (
            patch.dict(sys.modules, modules()),
            self.assertRaisesRegex(RuntimeError, "native sentinel"),
        ):
            fn(obj, layer, Tensor(), Tensor((3, 8)), Tensor((3, 8), "int32"), None)
        self.assertEqual(ops.calls, ["fused"])
        ops = Ops(fail_native=True)
        fn = load(ROUTER, "grouped_topk", {"torch": TORCH})
        with (
            patch.dict(sys.modules, modules(ops=ops)),
            self.assertRaisesRegex(RuntimeError, "native sentinel"),
        ):
            fn(
                Tensor(),
                Tensor(),
                8,
                True,
                1,
                1,
                "sigmoid",
                1.0,
                Tensor((256,), "fp32"),
            )
        self.assertEqual(ops.calls, ["router_native"])

    def test_static_input_scale_rejection(self):
        ops = Ops()
        obj, layer, fn = nn(ops)
        layer.w13_input_scale = Tensor()
        with self.assertRaisesRegex(NotImplementedError, "dynamic INT8"):
            fn(obj, layer, Tensor(), Tensor((3, 8)), Tensor((3, 8), "int32"), None)
        self.assertEqual(ops.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

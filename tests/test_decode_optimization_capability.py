# SPDX-License-Identifier: Apache-2.0
"""CPU branch regression: execute real selectors with explicit binary stand-ins.

No Torch/vLLM/ixformer imports, device discovery or GPU allocations. These tests
prove selection and error propagation, not native numerical correctness.
"""

import __future__

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    code = compile(
        ast.Module(body=[node], type_ignores=[]),
        path,
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, namespace)
    return namespace[name]


class Tensor:
    is_cuda = True
    dtype = "float32"
    device = "cuda:0"
    ndim = 2
    shape = (2, 256)

    def is_contiguous(self):
        return True

    def numel(self):
        return self.shape[0] * self.shape[1]


class CapabilityTest(unittest.TestCase):
    def run_selector(self, arch, selector, mode="radix", shape=(2, 256)):
        events = []
        opt = ModuleType("vllm.envs")
        arch_module = ModuleType("vllm_iluvatar.ilu_util.arch")
        exec(
            (ROOT / "vllm_iluvatar/ilu_util/arch.py").read_text(), arch_module.__dict__
        )
        opt.VLLM_ILU_GLM_SAMPLING_MODE = mode
        opt.VLLM_ILU_GLM_MOE_PREPARE_FUSION = True
        cuda = ModuleType("ixformer.cuda")
        cuda.cuda_arch = arch
        functions = ModuleType("ixformer.inference.functions")

        def native(*args, **kwargs):
            events.append("native")
            raise RuntimeError("native error must propagate")

        functions.moe_grouped_topk = lambda *a, **k: (
            events.append("original") or "weights",
            "ids",
        )
        functions.moe_sigmoid_topk = native
        functions.sampling_radix_top_p = native
        functions.sampling_sorted_top_p = native
        model = ModuleType("vllm.config.model")
        model.PROCESSED_LOGPROBS_MODES = ("processed_logprobs",)
        logger = ModuleType("vllm.logger")
        logger.init_logger = lambda *a: SimpleNamespace(info_once=lambda *a: None)
        fake_modules = {
            m.__name__: m for m in (opt, arch_module, cuda, functions, model, logger)
        }
        # Block parent package initialization too: importing a dotted alias
        # otherwise initializes the real ixformer even with its leaf mocked.
        for name in (
            "ixformer",
            "ixformer.inference",
            "vllm",
            "vllm.config",
            "vllm_iluvatar",
            "vllm_iluvatar.ilu_util",
        ):
            parent = ModuleType(name)
            parent.__path__ = []
            fake_modules[name] = parent
        for name, module in list(fake_modules.items()):
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(fake_modules[parent], child, module)
        torch = SimpleNamespace(float32="float32", bfloat16="bf16", float16="fp16")
        # A CQ selector is exercised with no explicit RNG noise. Non-CQ must
        # return before reaching either sort or the RNG allocation below.
        torch.empty_like = lambda *a: (_ for _ in ()).throw(
            AssertionError("RNG touched")
        )
        x = Tensor()
        x.shape = shape
        x.sort = lambda **k: (x, x)
        sampler = SimpleNamespace(use_fp64_gumbel=False, logprobs_mode="raw_logprobs")
        p = Tensor()
        p.ndim, p.shape = 1, (shape[0],)
        with patch.dict(sys.modules, fake_modules):
            self.assertEqual(arch_module.is_ivcore30(), arch == "ivcore30")
            if selector == "sampler":
                original = lambda *a: (events.append("original") or "tokens", None)
                fn = load_function(
                    "vllm_iluvatar/v1/sample/ops/topk_topp_sampler.py",
                    "forward_native",
                    {"torch": torch, "_original_forward_native": original},
                )
                args = (sampler, x, {}, None, p)
            else:
                fn = load_function(
                    "vllm_iluvatar/model_executor/layers/fused_moe/router/ilu_grouped_topk_router.py",
                    "grouped_topk",
                    {"torch": torch},
                )
                args = (x, x, 8, True, 1, 1, "sigmoid", 1.0, Tensor())
            expected_native = arch == "ivcore30" and (
                selector != "sampler"
                or (0 < shape[0] <= 65535 and 0 < x.numel() <= 2147483647)
            )
            if expected_native:
                with self.assertRaisesRegex(RuntimeError, "must propagate"):
                    fn(*args)
                self.assertEqual(events, ["native"])
            else:
                fn(*args)
                self.assertEqual(events, ["original"])
                # Mixed seeded batches must fall back before any RNG noise.
                if selector == "sampler":
                    fn(sampler, x, {0: object()}, None, p)
                    self.assertEqual(events, ["original", "original"])

    def test_binary_architecture_matrix(self):
        for arch in (None, "ivcore11", "ivcore40", "unknown", "ivcore30"):
            for selector in ("sampler", "router"):
                for mode in ("sorted", "radix"):
                    with self.subTest(arch=arch, selector=selector, mode=mode):
                        self.run_selector(arch, selector, mode)

    def test_sampling_native_size_limits(self):
        for shape in ((0, 256), (65536, 1), (65535, 32769)):
            with self.subTest(shape=shape):
                self.run_selector("ivcore30", "sampler", shape=shape)


class PreparationSelectionTest(unittest.TestCase):
    """Evaluate real pre-mutation predicates; no numerical fallback claim."""

    def predicate(self, path, selector, namespace):
        tree = ast.parse((ROOT / path).read_text())
        if selector == "standard":
            node = next(
                n.test
                for n in ast.walk(tree)
                if isinstance(n, ast.If) and "MOE_PREPARE" in ast.unparse(n.test)
            )
        else:
            node = next(
                n.value
                for n in ast.walk(tree)
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == selector for t in n.targets)
            )
        return eval(compile(ast.Expression(node), path, "eval"), namespace)

    def test_moe_padded_k_predicates(self):
        for selector, file in (
            ("fuse_quant", "nn_moe.py"),
            ("standard", "nn_experts.py"),
        ):
            path = (
                "vllm_iluvatar/model_executor/layers/quantization/"
                "backends/ivcore30/w4a8/" + file
            )
            for hidden, expected in ((192, False), (256, True), (8193, False)):
                x = Tensor()
                x.dtype, x.shape = "bf16", (2, hidden)
                ns = {
                    "envs": SimpleNamespace(VLLM_ILU_GLM_MOE_PREPARE_FUSION=True),
                    "is_ivcore30": lambda: True,
                    "input_scale": None,
                    "x": x,
                    "hidden_states": x,
                    "torch": SimpleNamespace(bfloat16="bf16"),
                    "self": SimpleNamespace(gemm1_info=SimpleNamespace(pad_k=256)),
                }
                with self.subTest(selector=selector, hidden=hidden):
                    self.assertEqual(self.predicate(path, selector, ns), expected)

    def test_dsa_constructor_heads(self):
        path = "vllm_iluvatar/models/glm_moe_dsa/attention.py"
        tree = ast.parse((ROOT / path).read_text())
        node = next(
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "_decode_prepare_enabled"
                for t in n.targets
            )
        )
        for heads in (8, 16, 32, 48, 64):
            a = SimpleNamespace(
                use_fp8_attn=True,
                num_heads=heads,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            )
            ns = {
                "envs": SimpleNamespace(VLLM_ILU_GLM_DSA_PREPARE_FUSION=True),
                "is_ivcore30": lambda: True,
                "self": a,
            }
            with self.subTest(heads=heads):
                self.assertEqual(
                    eval(compile(ast.Expression(node), path, "eval"), ns),
                    heads in (16, 32, 64),
                )

    def test_dsa_forward_fallback_predicates(self):
        path = "vllm_iluvatar/models/glm_moe_dsa/attention.py"
        reasons = (
            "eligible",
            "pcp",
            "unfrozen",
            "block",
            "dtype",
            "zero",
            "too_many",
            "cache",
            "weight",
            "disabled",
            "unknown_binary",
        )
        for reason in reasons:
            a = SimpleNamespace(
                _decode_prepare_enabled=reason != "disabled",
                _fp8_kv_scale_frozen=reason != "unfrozen",
            )
            q = SimpleNamespace(dtype="float" if reason == "dtype" else "bf16")
            ns = {
                "self": a,
                "use_pcp": reason == "pcp",
                "q": q,
                "is_ivcore30": lambda reason=reason: reason != "unknown_binary",
                "kv_rms_weight": None if reason == "weight" else object(),
                "num_actual_toks": (
                    0 if reason == "zero" else 65536 if reason == "too_many" else 17
                ),
                "attn_metadata": SimpleNamespace(
                    block_size=32 if reason == "block" else 64
                ),
                "kv_cache": SimpleNamespace(
                    numel=lambda reason=reason: 0 if reason == "cache" else 128
                ),
                "torch": SimpleNamespace(
                    bfloat16="bf16",
                    cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
                ),
            }
            with self.subTest(reason=reason):
                self.assertEqual(
                    self.predicate(path, "use_decode_prepare", ns), reason == "eligible"
                )

    def test_pcp_guard_mutation_is_detected(self):
        """Other conditions pass; removing only PCP exclusion hits the sentinel.

        This is a CPU predicate mutation proof, not a PCP forward test:
        the v0.31 branch does not contain the GLM PCP implementation.
        """
        import copy

        path = "vllm_iluvatar/models/glm_moe_dsa/attention.py"
        tree = ast.parse((ROOT / path).read_text())
        node = next(
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "use_decode_prepare"
                for t in n.targets
            )
        )
        ns = {
            "self": SimpleNamespace(
                use_fp8_attn=True,
                _decode_prepare_enabled=True,
                _fp8_kv_scale_frozen=True,
            ),
            "is_ivcore30": lambda: True,
            "use_pcp": True,
            "kv_rms_weight": object(),
            "q": SimpleNamespace(dtype="bf16"),
            "num_actual_toks": 1,
            "attn_metadata": SimpleNamespace(
                block_size=64,
                pcp_world_size=2,
                num_decode_tokens=1,
                slot_mapping=SimpleNamespace(numel=lambda: 2),
            ),
            "kv_cache": SimpleNamespace(numel=lambda: 40960),
            "torch": SimpleNamespace(
                bfloat16="bf16",
                cuda=SimpleNamespace(is_current_stream_capturing=lambda: False),
            ),
        }
        self.assertFalse(eval(compile(ast.Expression(node), path, "eval"), ns))

        class RemovePcpGuard(ast.NodeTransformer):
            def visit_UnaryOp(self, n):
                if (
                    isinstance(n.op, ast.Not)
                    and isinstance(n.operand, ast.Name)
                    and n.operand.id == "use_pcp"
                ):
                    return ast.copy_location(ast.Constant(True), n)
                return self.generic_visit(n)

        mutant = ast.fix_missing_locations(RemovePcpGuard().visit(copy.deepcopy(node)))
        self.assertTrue(eval(compile(ast.Expression(mutant), path, "eval"), ns))
        with self.assertRaisesRegex(AssertionError, "PCP selected local-only"):
            if eval(compile(ast.Expression(mutant), path, "eval"), ns):
                raise AssertionError("PCP selected local-only glm_dsa_prepare")


if __name__ == "__main__":
    unittest.main(verbosity=2)

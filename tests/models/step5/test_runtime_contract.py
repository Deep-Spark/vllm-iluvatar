"""Host-only Step5 startup contracts; runnable with stdlib unittest on macOS."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm_iluvatar import envs
from vllm_iluvatar.models.step5 import runtime_contract as contract


def configuration(*, tp=2, dp=1, ep=False, fp32_residual=True, sparse=True):
    return SimpleNamespace(
        device_config=SimpleNamespace(device_type="cuda"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=2,
                num_nextn_predict_layers=1,
                layer_types=["full_attention", "sliding_attention", "full_attention"],
                head_dim=192,
                fp32_residual_connection=fp32_residual,
                sparse_config={"enabled": sparse},
            )
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=dp,
            enable_expert_parallel=ep,
            all2all_backend="allgather_reducescatter",
        ),
        compilation_config=SimpleNamespace(mode=0, cudagraph_mode=0),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ["VLLM_STEP5_SPARSE_FP8"] = "1"
        os.environ["VLLM_STEP5_FP8_KV_SCALE_PATH"] = "frozen.json"

    def test_registered_switch_defaults_aliases_and_priority(self):
        for suffix, default in (
            ("SPARSE_ATTENTION", True),
            ("SPARSE_FP8", False),
            ("FP32_TP", False),
            ("FP32_EP", False),
        ):
            name, old = f"VLLM_ILU_STEP5_{suffix}", f"VLLM_STEP5_{suffix}"
            with self.subTest(suffix=suffix), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(getattr(envs, name), default)
                self.assertIs(envs._ILU_ENV_VARS[name], envs._ILU_ENV_ALIASES[old])
                os.environ[old] = "1"
                self.assertTrue(getattr(envs, name))
                os.environ[name] = "0"
                self.assertFalse(getattr(envs, name))
                self.assertFalse(getattr(envs, old))
                os.environ[name] = "invalid"
                with self.assertRaisesRegex(ValueError, name):
                    getattr(envs, name)
                del os.environ[name]
                os.environ[old] = "invalid"
                with self.assertRaisesRegex(ValueError, name):
                    getattr(envs, name)

    def test_registered_backend_and_scale_path_priority(self):
        for suffix, default, old_value, new_value in (
            ("FP8_ATTENTION_BACKEND", "ixformer_native", "ixattn", "ixformer_native"),
            ("FP8_KV_SCALE_PATH", "", "old.json", "new.json"),
        ):
            name, old = f"VLLM_ILU_STEP5_{suffix}", f"VLLM_STEP5_{suffix}"
            with self.subTest(suffix=suffix), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(getattr(envs, name), default)
                os.environ[old] = old_value
                self.assertEqual(getattr(envs, name), old_value)
                os.environ[name] = new_value
                self.assertEqual(getattr(envs, old), new_value)
        with (
            patch.dict(os.environ, {"VLLM_ILU_STEP5_FP8_ATTENTION_BACKEND": "typo"}),
            self.assertRaisesRegex(ValueError, "BACKEND"),
        ):
            _ = envs.VLLM_ILU_STEP5_FP8_ATTENTION_BACKEND

    def test_canonical_sparse_switch_is_used_by_startup(self):
        with patch.dict(
            os.environ,
            {
                "VLLM_STEP5_SPARSE_ATTENTION": "1",
                "VLLM_ILU_STEP5_SPARSE_ATTENTION": "0",
            },
        ):
            ops, _, queries = contract._native_requirements(
                configuration(), draft=False
            )
            self.assertNotIn("sparse_gqa_fp8", ops)
            self.assertEqual(queries, {})
        with (
            patch.dict(os.environ, {"VLLM_ILU_STEP5_SPARSE_FP8": "invalid"}),
            patch.object(contract, "_cuda_arch") as arch,
        ):
            with self.assertRaisesRegex(ValueError, "SPARSE_FP8"):
                contract.validate_step5_runtime(configuration())
            arch.assert_not_called()

    def test_required_operations_have_facades(self):
        for draft in (False, True):
            for sparse in (False, True):
                with self.subTest(draft=draft, sparse=sparse):
                    ops, defaults, queries = contract._native_requirements(
                        configuration(dp=8, ep=True, sparse=sparse), draft=draft
                    )
                    self.assertLessEqual(
                        set(ops) | set(queries), contract._FACADE_MODULES.keys()
                    )
                    self.assertLessEqual(defaults, set(ops))
                    self.assertEqual(bool(queries), sparse)

    def test_unsupported_communication_fails_before_extension_loading(self):
        for env, config, message in (
            ("VLLM_STEP5_FP32_TP", configuration(tp=2), "FP32 TP"),
            ("VLLM_STEP5_FP32_EP", configuration(dp=8, ep=True), "FP32 EP"),
        ):
            with (
                self.subTest(env=env),
                patch.dict(os.environ, {env: "1"}),
                patch.object(contract, "_cuda_arch") as arch,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    contract.validate_step5_runtime(config)
                arch.assert_not_called()

    def test_disabled_sparse_requires_no_sparse_operators(self):
        ops, _, queries = contract._native_requirements(
            configuration(sparse=False), draft=False
        )
        self.assertNotIn("sparse_gqa_fp8", ops)
        self.assertNotIn("top512", ops)
        self.assertEqual(queries, {})

    def test_sparse_requires_fp8_before_extension_loading(self):
        os.environ["VLLM_STEP5_SPARSE_FP8"] = "0"
        with patch.object(contract, "_cuda_arch") as arch:
            with self.assertRaisesRegex(ValueError, "requires sparse FP8"):
                contract.validate_step5_runtime(configuration())
            arch.assert_not_called()

    def test_nonzero_centered_rejected(self):
        config = configuration()
        config.model_config.hf_config.zero_centered = False
        with self.assertRaisesRegex(ValueError, "zero_centered"):
            contract.validate_step5_runtime(config)

    def test_cpu_reference_does_not_load_native_runtime(self):
        config = configuration()
        config.device_config.device_type = "cpu"
        with patch.object(contract, "_native_runtime") as runtime:
            self.assertEqual(contract.validate_step5_runtime(config), None)
            runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()

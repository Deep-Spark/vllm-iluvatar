# SPDX-License-Identifier: Apache-2.0
"""Real CPU tensors + actual forward/FP8 absorb methods; native is a stand-in.

Proves layout/configuration integration and copy scope, not GPU numerics.
"""

import ast
import __future__
import os
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = "vllm_iluvatar/models/glm_moe_dsa/attention.py"
FP8 = "vllm_iluvatar/models/glm_moe_dsa/fp8_attention.py"
ABSORB = "VLLM_ILU_GLM_DSA_FP8_ABSORB_UP"
PREPARE = "VLLM_ILU_GLM_DSA_PREPARE_FUSION"


def method(path, cls, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(
        compile(
            ast.Module(body=[fn], type_ignores=[]),
            path,
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[name]


class PrepareLayoutCPU(unittest.TestCase):
    def run_forward(self, tokens, fp8_absorb, prepare_enabled):
        heads, nope, lora, value = 32, 192, 512, 256
        events = []
        projected = []
        config = {}
        exec((ROOT / "vllm_iluvatar/envs.py").read_text(), config)
        readers = config["_ILU_CANONICAL_ONLY_ENV_VARS"]
        metadata = SimpleNamespace(
            num_actual_tokens=tokens,
            pcp_world_size=1,
            block_size=64,
            slot_mapping=torch.arange(tokens),
            num_decode_tokens=0,
        )

        def prepare(q_nope, q_pe, *args, **kwargs):
            for q in (q_nope, q_pe):
                self.assertEqual(q.stride(2), 1)
                self.assertGreaterEqual(q.stride(1), q.size(2))
                self.assertGreaterEqual(q.stride(0), heads * q.stride(1))
            self.assertEqual(q_nope.stride(0) % 8, 0)
            self.assertEqual(q_nope.stride(1) % 8, 0)
            original = projected[-1]
            incompatible = original.stride(0) < heads * original.stride(1)
            self.assertEqual(q_nope.data_ptr() != original.data_ptr(), incompatible)
            self.assertTrue(torch.equal(q_nope, original))
            events.append("prepare")
            return ("packed", "scale")

        def rms_norm(x, w, eps):
            events.append("norm")
            return x

        def fp8_einsum(a, weight, a_scale, weight_scale):
            # Compatible padded token-major native output stand-in.
            out = torch.zeros((tokens, heads, lora + 8), dtype=torch.bfloat16)
            return out[:, :, :lora]

        ops = SimpleNamespace(
            glm_dsa_prepare=prepare,
            rms_norm=rms_norm,
            glm_dsa_fp8_kv_insert=lambda *a, **k: events.append("insert"),
            alloc_readsafe_scale=lambda h, n, device: torch.empty(
                (h, n), dtype=torch.float32, device=device
            ),
            fp8_grouped_per_token_quant=lambda a, h: (a, None),
            fp8_per_channel_einsum=fp8_einsum,
        )
        namespace = {
            "torch": torch,
            "ixf_ops": ops,
            "get_forward_context": lambda: SimpleNamespace(attn_metadata=metadata),
            "_FP8_MAX": 448.0,
            "_FP8_DTYPE": torch.float8_e4m3fn,
            "IluGlmMoeDsaFp8SparseMLA": SimpleNamespace(_absorb_logged=True),
        }
        forward = method(BASE, "IluGlmMoeDsaSparseMLA", "forward", namespace)
        cls = "IluGlmMoeDsaFp8SparseMLA"
        obj = SimpleNamespace(
            num_heads=heads,
            kv_lora_rank=lora,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=64,
            v_head_dim=value,
            prefix="test",
            _fp8_kv_scale_frozen=True,
            use_rms_rope_fp8_kv_fusion=False,
            use_fp8_attn=True,
            kv_cache=torch.zeros((1, 64, 640), dtype=torch.uint8),
            _k_scale=torch.ones(1),
            kv_b_proj=SimpleNamespace(weight=True),
        )
        with patch.dict(
            os.environ,
            {ABSORB: str(int(fp8_absorb)), PREPARE: str(int(prepare_enabled))},
        ):
            obj.use_fp8_absorb_up = readers[ABSORB]()
            obj._decode_prepare_enabled = readers[PREPARE]()
        obj._read_kv_b_weight = lambda: (
            torch.ones((heads, nope + value, lora)).to(torch.float8_e4m3fn),
            torch.ones((heads, nope + value)),
            (heads, nope, value, lora),
        )
        for name in ("_build_absorb", "_build_bf16_absorb", "_build_fp8_absorb"):
            setattr(obj, name, MethodType(method(FP8, cls, name, namespace), obj))
        obj._build_absorb()
        self.assertEqual(obj.W_UK_S is not None, fp8_absorb)
        actual_k_up = MethodType(method(FP8, cls, "_k_up_proj", namespace), obj)

        def k_up(q):
            result = actual_k_up(q)
            projected.append(result)
            return result

        obj._k_up_proj = k_up

        class Rotary:
            def __call__(self, positions, q, k):
                events.append("rope")
                return q, k

            def _match_cos_sin_cache_dtype(self, q):
                return torch.zeros((tokens, 64), dtype=q.dtype)

        obj.rotary_emb = Rotary()
        obj._maybe_init_fp8_kv_scale = lambda *a: None

        def sparse(q_nope, q_pe, cache, meta, prepared=None):
            self.assertEqual(prepared is not None, prepare_enabled)
            if not prepare_enabled:
                self.assertEqual(q_nope.data_ptr(), projected[-1].data_ptr())
                self.assertEqual(q_nope.stride(2), 1)  # old pack accepts this
            events.append("prepared_sparse" if prepared is not None else "old_pack")
            return torch.zeros((tokens, heads, lora), dtype=torch.bfloat16)

        obj._sparse_mla_kernel_fp8 = sparse
        obj._v_up_proj = lambda x, **k: torch.zeros(
            (tokens, heads, value), dtype=torch.bfloat16
        )
        arch = ModuleType("vllm_iluvatar.ilu_util.arch")
        arch.is_ivcore30 = lambda: True
        parents = {
            n: ModuleType(n) for n in ("vllm_iluvatar", "vllm_iluvatar.ilu_util")
        }
        for module in parents.values():
            module.__path__ = []
        with patch.dict(sys.modules, {**parents, arch.__name__: arch}):
            q = (
                torch.arange(tokens * heads * (nope + 64))
                .remainder(13)
                .reshape(tokens, heads, nope + 64)
                .to(torch.bfloat16)
            )
            kv = torch.zeros((tokens, lora + 64), dtype=torch.bfloat16)
            output = forward(
                obj,
                q,
                kv[:, :lora],
                kv[:, lora:],
                torch.arange(tokens),
                torch.ones(lora, dtype=torch.bfloat16),
                1e-5,
            )
        self.assertEqual(output.shape, (tokens, heads * value))
        if not fp8_absorb and tokens > 1:
            self.assertEqual(projected[0].stride(), (512, tokens * 512, 1))
        self.assertEqual("prepare" in events, prepare_enabled)
        self.assertEqual("norm" in events, not prepare_enabled)
        self.assertEqual("rope" in events, not prepare_enabled)
        self.assertFalse(torch.cuda.is_initialized())

    def test_bf16_absorb_prepare_configuration_matrix(self):
        for tokens in (1, 4):
            for enabled in (False, True):
                with self.subTest(tokens=tokens, prepare=enabled):
                    self.run_forward(tokens, False, enabled)

    def test_compatible_padded_fp8_output_is_not_copied(self):
        for enabled in (False, True):
            with self.subTest(prepare=enabled):
                self.run_forward(4, True, enabled)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)

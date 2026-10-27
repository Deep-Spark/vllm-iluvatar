# SPDX-License-Identifier: Apache-2.0
"""Meta SymInt contracts + one lightweight compile numerical check.

Regression for SWAPP-1833: ``int64_t n = t.size(0); empty({n, ...})`` specializes
the Dynamo ``mark_dynamic`` token dim (e.g. to 2048) and breaks
FULL_AND_PIECEWISE ``build_guards``.

Most cases assert FakeTensor / Meta keep token dims symbolic. One case compiles
``minimax_qk_var`` with ``mark_dynamic`` and checks eager==compiled numerically
(representative; other Meta ops share the same Sym pattern).
"""

from __future__ import annotations

import pytest
import torch
import vllm_iluvatar._C  # noqa: F401
from conftest import requires_cuda
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

# Meta dispatch uses device=meta; _C still needs a built Iluvatar wheel.
pytestmark = [requires_cuda]


def _sym_empty(
    shape_env: ShapeEnv,
    *dims: int | None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "meta",
) -> torch.Tensor:
    sizes: list[int | torch.SymInt] = [
        shape_env.create_unbacked_symint() if d is None else d for d in dims
    ]
    return torch.ops.aten.empty.memory_format(
        sizes,
        dtype=dtype,
        device=torch.device(device),
        pin_memory=False,
        memory_format=torch.contiguous_format,
    )


def _assert_token_dim_symbolic(t: torch.Tensor, *, name: str) -> None:
    s0 = t.shape[0]
    assert isinstance(s0, torch.SymInt), (
        f"{name}: expected SymInt token dim, got {type(s0).__name__}={s0!r}"
    )
    assert not s0.node.expr.is_number, (
        f"{name}: token dim specialized to constant {s0!r} "
        f"(Meta must use sym_size/empty_symint, not size(0)+empty)"
    )


@torch.inference_mode()
def test_minimax_qk_var_meta_keeps_token_symint() -> None:
    shape_env = ShapeEnv()
    q_size, kv_size = 128, 64
    with FakeTensorMode(shape_env=shape_env):
        qkv = _sym_empty(shape_env, None, q_size + 2 * kv_size)
        out = torch.ops._C.minimax_qk_var(qkv, q_size, kv_size)
        _assert_token_dim_symbolic(out, name="minimax_qk_var")
        assert out.shape[1] == 2
        assert out.dtype == torch.float32


@torch.inference_mode()
def test_minimax_qk_var_compile_mark_dynamic_matches_eager() -> None:
    """One-op smoke: compile + mark_dynamic(token) still matches eager."""
    torch.manual_seed(0)
    q_size, kv_size = 128, 64
    dtype = torch.bfloat16

    def fn(qkv: torch.Tensor) -> torch.Tensor:
        return torch.ops._C.minimax_qk_var(qkv, q_size, kv_size)

    compiled = torch.compile(fn, fullgraph=True, dynamic=True)

    def _run(num_tokens: int) -> None:
        qkv = torch.randn(
            num_tokens, q_size + 2 * kv_size, dtype=dtype, device="cuda"
        )
        torch._dynamo.mark_dynamic(qkv, 0)
        eager = fn(qkv)
        out = compiled(qkv)
        torch.testing.assert_close(out, eager, atol=0, rtol=0)

    _run(7)
    _run(11)  # different token count after first compile


@torch.inference_mode()
def test_minimax_qk_rms_apply_meta_keeps_token_symint() -> None:
    shape_env = ShapeEnv()
    q_size, kv_size = 128, 64
    with FakeTensorMode(shape_env=shape_env):
        qkv = _sym_empty(shape_env, None, q_size + 2 * kv_size)
        qk_var = _sym_empty(shape_env, None, 2, dtype=torch.float32)
        q_weight = torch.empty(q_size, dtype=torch.bfloat16, device="meta")
        k_weight = torch.empty(kv_size, dtype=torch.bfloat16, device="meta")
        q_out, k_out = torch.ops._C.minimax_qk_rms_apply(
            qkv, qk_var, q_weight, k_weight, q_size, kv_size, 1, 1e-6
        )
        _assert_token_dim_symbolic(q_out, name="minimax_qk_rms_apply.q")
        _assert_token_dim_symbolic(k_out, name="minimax_qk_rms_apply.k")
        assert q_out.shape[1] == q_size
        assert k_out.shape[1] == kv_size


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_qk_rmsnorm_gate_meta_keeps_token_symint(dtype: torch.dtype) -> None:
    shape_env = ShapeEnv()
    num_q_heads, num_kv_heads, head_dim = 2, 1, 128
    with FakeTensorMode(shape_env=shape_env):
        q_gate = _sym_empty(shape_env, None, num_q_heads * 2 * head_dim, dtype=dtype)
        k = _sym_empty(shape_env, None, num_kv_heads * head_dim, dtype=dtype)
        q_weight = torch.empty(head_dim, dtype=dtype, device="meta")
        k_weight = torch.empty(head_dim, dtype=dtype, device="meta")
        q_out, k_out, gate_out = torch.ops._C.fused_qk_rmsnorm_gate(
            q_gate,
            k,
            q_weight,
            k_weight,
            1e-6,
            num_q_heads,
            num_kv_heads,
            head_dim,
        )
        _assert_token_dim_symbolic(q_out, name="fused_qk_rmsnorm_gate.q")
        _assert_token_dim_symbolic(k_out, name="fused_qk_rmsnorm_gate.k")
        _assert_token_dim_symbolic(gate_out, name="fused_qk_rmsnorm_gate.gate")
        assert q_out.shape[1] == num_q_heads * head_dim
        assert k_out.shape[1] == num_kv_heads * head_dim
        assert gate_out.shape[1] == num_q_heads * head_dim


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_qk_rmsnorm_rope_gate_meta_keeps_token_symint(
    dtype: torch.dtype,
) -> None:
    shape_env = ShapeEnv()
    num_q_heads, num_kv_heads, head_dim = 2, 1, 128
    with FakeTensorMode(shape_env=shape_env):
        q_gate = _sym_empty(shape_env, None, num_q_heads * 2 * head_dim, dtype=dtype)
        k = _sym_empty(shape_env, None, num_kv_heads * head_dim, dtype=dtype)
        q_weight = torch.empty(head_dim, dtype=dtype, device="meta")
        k_weight = torch.empty(head_dim, dtype=dtype, device="meta")
        cos_sin_cache = torch.empty(32, head_dim, dtype=torch.float32, device="meta")
        positions = _sym_empty(shape_env, None, dtype=torch.int64)
        q_out, k_out, gate_out = torch.ops._C.fused_qk_rmsnorm_rope_gate(
            q_gate,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            1e-6,
            num_q_heads,
            num_kv_heads,
            head_dim,
            head_dim,
        )
        _assert_token_dim_symbolic(q_out, name="fused_qk_rmsnorm_rope_gate.q")
        _assert_token_dim_symbolic(k_out, name="fused_qk_rmsnorm_rope_gate.k")
        _assert_token_dim_symbolic(gate_out, name="fused_qk_rmsnorm_rope_gate.gate")
        assert q_out.shape[1] == num_q_heads * head_dim
        assert k_out.shape[1] == num_kv_heads * head_dim
        assert gate_out.shape[1] == num_q_heads * head_dim

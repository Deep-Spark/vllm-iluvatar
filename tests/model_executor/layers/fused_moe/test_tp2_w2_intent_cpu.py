"""Execute actual plugin control flow with Python doubles, without GPU imports."""
import ast
from pathlib import Path
import types
import unittest

ROOT = Path(__file__).resolve().parents[4]


def method(path, name):
    fn = next(n for n in ast.walk(ast.parse(path.read_text()))
              if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, fn], type_ignores=[]))
    return fn, compile(code, str(path), 'exec')


class Tensor:
    def __init__(self, rows, width=5120): self.shape = (rows, width)
    def __getitem__(self, rows):
        return Tensor(len(range(*rows.indices(self.shape[0]))), self.shape[1])
    def view(self, shape): return self


class AtPrepare(Exception): pass


class Intent(unittest.TestCase):
    def test_actual_scatter_and_finalizer_choice(self):
        path = ROOT / 'vllm_iluvatar/model_executor/layers/fused_moe/ilu_fused_moe_ep.py'
        fn, code = method(path, '_forward_dp_ep_flatten')
        for tp, fused, scatter in ((1, True, False), (2, False, False),
                                   (2, False, True), (2, True, False), (2, True, True)):
            for rank in range(tp):
                for rows in (0, 1, 5, 6, 12, 17, 18, 24):
                    observed = []
                    def prepare(hidden, weights, ids, *args, **kwargs):
                        observed.append((hidden.shape[0], weights.shape[0], ids.shape[0], kwargs))
                        raise AtPrepare
                    pf = types.SimpleNamespace(supports_async=lambda:True, prepare_async=prepare)
                    experts = types.SimpleNamespace(quant_config=object(), expects_unquantized_inputs=False)
                    kernel = types.SimpleNamespace(prepare_finalize=pf, fused_experts=experts)
                    routed = types.SimpleNamespace(w13_weight=object(), w2_weight=object(),
                                                   global_num_experts=384, expert_map=None,
                                                   apply_router_weight_on_input=False)
                    namespace = dict(get_tensor_model_parallel_world_size=lambda:tp,
                                     get_tensor_model_parallel_rank=lambda:rank,
                                     VLLM_ILU_LL_FUSED_AG_AR=fused,
                                     VLLM_ILU_DEEPEP_USE_SCATTER=scatter)
                    exec(code, namespace)
                    owner = types.SimpleNamespace(experts=types.SimpleNamespace(routed_experts=routed))
                    with self.assertRaises(AtPrepare):
                        namespace[fn.name](owner, Tensor(rows), Tensor(rows,6), Tensor(rows,6),
                                           None, moe_kernel=kernel)
                    local = ((rows // tp) + int(rank < rows % tp)
                             if tp > 1 and (scatter or fused) else rows)
                    n, w, i, kw = observed[0]
                    self.assertEqual((n,w,i), (local,local,local))
                    self.assertEqual(kw.get('w2_direct_finalize'), 'tp_fused' if tp == 2 and fused else None)
                    self.assertIs(kw['defer_input_quant'], False)

    def test_tail_prefill_does_not_enter_decode_helper(self):
        path = ROOT / 'vllm_iluvatar/models/deepseek_v4/fp8_model.py'
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == 'IluDeepseekV4Fp8MoE')
        fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == 'forward')
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,fn],type_ignores=[])),str(path),'exec'),namespace)
        for rows in (128,129):
            calls=[]
            owner = types.SimpleNamespace(gate=types.SimpleNamespace(tid2eid=None),image_sentinel_lo=0,
                _ilu_decode_ll=True, _is_all_ranks_decode_only=lambda:False,
                forward_dp_ep_deep_ep=lambda *a:self.fail('tail entered decode direct chain'),
                experts=lambda **kw:(calls.append('ordinary') or Tensor(rows)))
            namespace['forward'](owner,Tensor(rows))
            self.assertEqual(calls,['ordinary'])


if __name__ == '__main__': unittest.main()

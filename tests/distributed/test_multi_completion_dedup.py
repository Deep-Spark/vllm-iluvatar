"""CPU contracts for distinct child/worker completions using pinned methods.

GPU completion signals are test inputs; this does not simulate device safety.
The inherited source tests retain their original parent and Nixl method paths.
"""
from __future__ import annotations

import importlib.util
import ast
import hashlib
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
import unittest


path = Path(__file__).with_name("test_multi_completion_sources.py")
spec = importlib.util.spec_from_file_location("multi_completion_original_tests", path)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


class DistinctCompletionTests(base.MultiCompletionTests):
    def setUp(self):
        super().setUp()
        self.GuardedAggregator = type("DistinctAggregator", (self.Aggregator,), {})
        self.module.install_multi_rank_completion_dedup(self.GuardedAggregator)

    def output(self, sends=(), sources=(0, 1), recvs=(), failed=(), invalid=(), expected=0):
        meta = self.module.MultiCompletionWorkerMetadata(
            (None, None), {rid: set(sources) for rid in sends},
            {rid: {1} for rid in recvs},
        )
        value = base.Transfer(set(sends), set(recvs), set(failed), set(invalid),
                              kv_connector_worker_meta=meta, expected_finished_count=expected)
        return SimpleNamespace(kv_connector_output=value, other_payload=object())

    def test_repeat_child_does_not_replace_other_child_after_extra_counter_removed(self):
        for reverse in (False, True):
            nx, lc = self.child(), self.child()
            nx.reports = [base.Transfer({'R'}), base.Transfer({'R'}), base.Transfer()]
            lc.reports = [base.Transfer(), base.Transfer(), base.Transfer({'R'})]
            worker = self.multi([lc, nx] if reverse else [nx, lc])
            worker._extra_async_saves = {'R': 1}
            self.assertFalse(self.frame(worker).finished_sending)
            self.assertNotIn('R', worker._extra_async_saves)
            self.assertFalse(self.frame(worker).finished_sending)
            last = self.frame(worker)
            self.assertEqual(last.finished_sending, {'R'})
            self.assertEqual(last.kv_connector_worker_meta.send_sources, {'R': {0, 1}})
            self.assertFalse(worker._ilu_multi_worker_send_sources)

    def test_eight_workers_repeated_first_worker_cannot_replace_last_worker(self):
        for output_rank in (0, 7):
            workers = [self.multi([self.child(), self.child()]) for _ in range(8)]
            for worker in workers:
                worker._extra_async_saves = {'R': 1}
                for child in worker._connectors:
                    child.reports.clear()
            nx = self.child(nixl=True, pending={'R'}, awaiting={'R'})
            scheduler = self.multi([nx, self.child()])
            parent, freed = self.parent(scheduler, ['R'])
            aggregator = self.GuardedAggregator(8)

            def step(events):
                for rank, children in events.items():
                    for child in children:
                        workers[rank]._connectors[child].reports.append(base.Transfer({'R'}))
                frames = [SimpleNamespace(kv_connector_output=self.frame(w)) for w in workers]
                result = aggregator.aggregate(frames, output_rank=output_rank)
                parent._update_from_kv_xfer_finished(result.kv_connector_output)

            step({0: (0, 1)})
            self.assertEqual(aggregator._send_remaining_count, {'R': 7})
            step({i: (0, 1) for i in range(1, 7)})
            self.assertEqual(aggregator._send_remaining_count, {'R': 1})
            step({0: (0,)})
            self.assertEqual(aggregator._send_remaining_count, {'R': 1})
            self.assertFalse(freed)
            step({7: (0, 1)})
            self.assertEqual(freed, ['R'])
            self.assertFalse(aggregator._send_remaining_count)
            self.assertFalse(aggregator._ilu_multi_send_worker_positions)
            # A later protocol terminal retry still reaches the old Nixl
            # stale filter. It does not create a second parent free.
            step({i: (0,) for i in range(8)})
            self.assertEqual(freed, ['R'])

    def test_none_output_rank_preserves_pending_barrier(self):
        aggregator = self.GuardedAggregator(8)
        aggregator.aggregate([self.output({'R'})] + [self.output() for _ in range(7)], 7)
        self.assertEqual(aggregator._send_remaining_count, {'R': 7})
        frames = [self.output({'R'})] + [self.output() for _ in range(6)] + [None]
        self.assertIsNone(aggregator.aggregate(frames, 7))
        self.assertEqual(aggregator._send_remaining_count, {'R': 7})
        self.assertEqual(aggregator._ilu_multi_send_worker_positions, {'R': {0}})

    def test_input_shells_and_completion_metadata_remain_unchanged(self):
        aggregator = self.GuardedAggregator(2)
        first = self.output({'R'})
        frames = [first, self.output()]
        kv = first.kv_connector_output
        meta = kv.kv_connector_worker_meta
        result = aggregator.aggregate(frames)
        self.assertIs(first.kv_connector_output, kv)
        self.assertIs(kv.kv_connector_worker_meta, meta)
        self.assertEqual(kv.finished_sending, {'R'})
        self.assertEqual(meta.send_sources, {'R': {0, 1}})
        self.assertIs(result.other_payload, first.other_payload)
        self.assertFalse(result.kv_connector_output.finished_sending)

    def test_receive_failure_invalid_blocks_and_expected_count_match_original(self):
        def inputs():
            return [self.output({'S'}, recvs={'R'}, failed={'R'}, invalid={i}, expected=2)
                    for i in range(2)]
        ordinary = self.Aggregator(8).aggregate(inputs(), 1).kv_connector_output
        guarded = self.GuardedAggregator(8).aggregate(inputs(), 1).kv_connector_output
        for name in ('finished_sending', 'finished_recving', 'failed_recving',
                     'invalid_block_ids', 'expected_finished_count'):
            self.assertEqual(getattr(guarded, name), getattr(ordinary, name), name)

    def test_existing_barrier_uses_original_remaining_when_default_count_changes(self):
        aggregator = self.GuardedAggregator(3)
        aggregator.aggregate([self.output({'R'}), self.output(), self.output()])
        self.assertEqual(aggregator._send_remaining_count, {'R': 2})
        frames = [self.output(expected=8), self.output({'R'}), self.output()]
        self.assertFalse(aggregator.aggregate(frames).kv_connector_output.finished_sending)
        self.assertEqual(aggregator._send_remaining_count, {'R': 1})
        result = aggregator.aggregate([self.output(), self.output(), self.output({'R'})])
        self.assertEqual(result.kv_connector_output.finished_sending, {'R'})
        self.assertEqual(result.kv_connector_output.expected_finished_count, 8)

    def test_plain_non_multi_output_keeps_original_behavior(self):
        aggregator = self.GuardedAggregator(2)
        def inputs():
            return [SimpleNamespace(kv_connector_output=base.Transfer({'R'})),
                    SimpleNamespace(kv_connector_output=base.Transfer())]
        self.assertFalse(aggregator.aggregate(inputs()).kv_connector_output.finished_sending)
        self.assertEqual(aggregator.aggregate(inputs()).kv_connector_output.finished_sending, {'R'})
        self.assertFalse(hasattr(aggregator, '_ilu_multi_send_worker_positions'))

    def test_rank_installer_is_idempotent(self):
        original = self.GuardedAggregator.aggregate
        self.module.install_multi_rank_completion_dedup(self.GuardedAggregator)
        self.assertIs(self.GuardedAggregator.aggregate, original)

    def test_rank_method_source_guard_matches_pinned_actual_method(self):
        registration = base.REPO / 'vllm_iluvatar/ilu_patch/general_patches/patch_multi_completion_sources.py'
        tree = ast.parse(registration.read_text(encoding='utf-8'))
        expected = next(ast.literal_eval(node.value) for node in tree.body
                        if isinstance(node, ast.Assign) and any(
                            isinstance(target, ast.Name) and target.id == '_AGGREGATE_FINGERPRINT'
                            for target in node.targets))
        source = self.vllm / 'distributed/kv_transfer/kv_connector/utils.py'
        lines = source.read_text(encoding='utf-8').splitlines()
        cls = next(node for node in ast.parse('\n'.join(lines)).body
                   if isinstance(node, ast.ClassDef) and node.name == 'KVOutputAggregator')
        method = next(node for node in cls.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'aggregate')
        actual = textwrap.dedent('\n'.join(lines[method.lineno - 1:method.end_lineno])).strip()
        self.assertEqual(hashlib.sha256(actual.encode()).hexdigest(), expected)

    def test_actual_loader_body_rejects_every_partial_marker_combination(self):
        registration = base.REPO / 'vllm_iluvatar/ilu_patch/general_patches/patch_multi_completion_sources.py'
        tree = ast.parse(registration.read_text(encoding='utf-8'))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == '_apply')
        # Replace imports by explicit CPU class inputs; execute the actual
        # registration decisions, not a second implementation of the guard.
        function.body = [node for node in function.body
                         if not isinstance(node, (ast.Import, ast.ImportFrom))]
        compiled = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                           str(registration), 'exec')
        names = ('get_transfer_results', 'build_connector_worker_meta', 'update_connector_output')
        for bits in range(1, 32):
            with self.subTest(marker_mask=bits):
                multi = type('LoaderMulti', (), {name: (lambda self: None) for name in names})
                agg = type('LoaderAggregate', (), {'aggregate': lambda self: None})
                marked = [(getattr(multi, name), '_ilu_multi_completion_sources') for name in names]
                marked += [(multi.get_transfer_results, '_ilu_multi_distinct_child_sends'),
                           (agg.aggregate, '_ilu_multi_distinct_worker_sends')]
                for index, (method, name) in enumerate(marked):
                    if bits & (1 << index):
                        setattr(method, name, True)
                scope = {'MultiConnector': multi, 'KVOutputAggregator': agg,
                         '_FINGERPRINTS': dict.fromkeys(names)}
                exec(compiled, scope)
                if bits == 31:
                    scope['_apply']()
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Partial or older'):
                        scope['_apply']()


if __name__ == '__main__':
    unittest.main(verbosity=2)

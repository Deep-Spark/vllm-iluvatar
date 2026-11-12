"""CPU actual-method contracts; no model/native import or GPU is required.

Set VLLM_SOURCE_ROOT to the vllm package directory when testing a source tree.
Installed distributions are located without importing their model/backend.
"""
from __future__ import annotations
import ast
from contextlib import suppress
from dataclasses import dataclass, field
import gc
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import sys
import textwrap
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


REPO = Path(__file__).resolve().parents[2]
HELPER = REPO/'vllm_iluvatar/distributed/kv_transfer/kv_connector/v1/multi_completion.py'


def source_package(name):
    explicit = os.environ.get(name.upper()+'_SOURCE_ROOT')
    if explicit:
        return Path(explicit)
    try:
        return Path(importlib.metadata.distribution(name).locate_file(name))
    except importlib.metadata.PackageNotFoundError:
        raise unittest.SkipTest('An installed '+name+' or explicit source root is required')


@dataclass
class Transfer:
    finished_sending: set | None = field(default_factory=set)
    finished_recving: set | None = field(default_factory=set)
    failed_recving: set = field(default_factory=set)
    invalid_block_ids: set = field(default_factory=set)
    kv_connector_worker_meta: object = None
    kv_connector_stats: object = None
    kv_cache_events: object = None
    expected_finished_count: int = 0


class Logger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


STATUS = SimpleNamespace(FINISHED_STOPPED=1, FINISHED_LENGTH_CAPPED=2, WAITING_FOR_REMOTE_KVS=3,
                         is_finished=lambda value: value in (1,2))


def definitions(path, class_name, names, namespace):
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==class_name)
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    nodes += [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope=dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
    return {name:scope[name] for name in names}


class MultiCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vllm=source_package('vllm')
        cls.multi_file=cls.vllm/'distributed/kv_transfer/kv_connector/v1/multi_connector.py'
        nx=os.environ.get('NIXL_SCHEDULER_SOURCE')
        cls.nixl_file=Path(nx) if nx else REPO/'vllm_iluvatar/distributed/kv_transfer/kv_connector/v1/opt_nixl/scheduler.py'
        if not cls.nixl_file.exists():raise unittest.SkipTest('Pinned Nixl scheduler source is required')
        cls.stub_request=ModuleType('vllm.v1.request');cls.stub_request.RequestStatus=STATUS
        cls.meta_module=ModuleType('multi_source_actual_metadata');sys.modules[cls.meta_module.__name__]=cls.meta_module
        ns=cls.meta_module.__dict__
        ns.update(dataclass=dataclass,KVConnectorWorkerMetadata=object)
        tree=ast.parse(cls.multi_file.read_text())
        meta=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiKVConnectorWorkerMetadata')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[meta],type_ignores=[])),str(cls.multi_file),'exec'),ns)
        cls.Meta=ns['MultiKVConnectorWorkerMetadata']
        namespace=dict(KVConnectorTransferResults=Transfer,MultiKVConnectorWorkerMetadata=cls.Meta,
                       logger=Logger(),suppress=suppress,RequestStatus=STATUS)
        cls.original=definitions(cls.multi_file,'MultiConnector',
            ['get_transfer_results','build_connector_worker_meta','update_connector_output'],namespace)
        cls.nx_methods=definitions(cls.nixl_file,'NixlConnectorScheduler',
            ['update_connector_output'],namespace)
        sched=cls.vllm/'v1/core/sched/scheduler.py'
        cls.parent_methods=definitions(sched,'Scheduler',['_update_from_kv_xfer_finished','_free_blocks',
            '_free_request_blocks','_request_blocks_can_be_freed'],namespace)
        ns=dict(namespace,KVConnectorOutput=Transfer)
        utils=cls.vllm/'distributed/kv_transfer/kv_connector/utils.py'
        tree=ast.parse(utils.read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='KVOutputAggregator')
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[])),str(utils),'exec'),ns)
        cls.Aggregator=ns['KVOutputAggregator']

    def setUp(self):
        base=ModuleType('vllm.distributed.kv_transfer.kv_connector.v1.base')
        base.KVConnectorTransferResults=Transfer
        multi=ModuleType('vllm.distributed.kv_transfer.kv_connector.v1.multi_connector')
        multi.MultiKVConnectorWorkerMetadata=self.Meta
        self.modules=patch.dict(sys.modules,{
            base.__name__:base,multi.__name__:multi,'vllm.v1.request':self.stub_request})
        self.modules.start();self.addCleanup(self.modules.stop)
        name='candidate_multi_completion_sources'
        spec=importlib.util.spec_from_file_location(name,HELPER)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        self.module=module
        self.Multi=type('ActualCpuMulti',(),dict(self.original))
        module.install_multi_completion_sources(self.Multi)

    def multi(self,children):
        result=self.Multi();result._connectors=children;result._extra_async_saves={}
        return result

    def child(self,sends=(),recvs=(),nixl=False,pending=(),awaiting=()):
        state=SimpleNamespace(prefix_state=None,_scheduler_failed_recvs=set(),_live_recv_reqs=set(),
            _recv_release_proofs=set(),_ownership_generations={},_pending_worker_sends=set(pending),
            _sends_awaiting_release=set(awaiting),_sends_completed_early=set(),_stop_recv_tracking=lambda rid:None)
        state.update_connector_output=MethodType(self.nx_methods['update_connector_output'],state)
        child=SimpleNamespace(connector_scheduler=state if nixl else None,
            reports=[Transfer(set(sends),set(recvs))],seen=[])
        def results(ids):return child.reports.pop(0) if child.reports else Transfer()
        def update(output):
            child.seen.append((set(output.finished_sending or ()),set(output.finished_recving or ()),
                               set(output.failed_recving or ()),output.kv_connector_worker_meta))
            if nixl:state.update_connector_output(output)
        child.get_transfer_results=results;child.update_connector_output=update
        child.build_connector_worker_meta=lambda:None
        return child

    def frame(self,worker):
        value=worker.get_transfer_results(set())
        value.kv_connector_worker_meta=worker.build_connector_worker_meta()
        return value

    def parent(self,connector,ids):
        freed=[]
        requests={rid:SimpleNamespace(request_id=rid,status=1,is_finished=lambda:True) for rid in ids}
        parent=SimpleNamespace(connector=connector,requests=requests,
            kv_cache_manager=SimpleNamespace(free=lambda req:freed.append(req.request_id)),
            defer_block_free=False,finished_recving_kv_req_ids=set())
        for name,method in self.parent_methods.items():setattr(parent,name,MethodType(method,parent))
        return parent,freed

    def test_legal_lmcache_completion_both_orders_reaches_actual_parent_free(self):
        for reverse in (False,True):
            lc,nx=self.child({'L'}),self.child(nixl=True)
            children=[nx,lc] if reverse else [lc,nx]
            connector=self.multi(children);frame=self.frame(connector)
            parent,freed=self.parent(connector,['L']);parent._update_from_kv_xfer_finished(frame)
            self.assertEqual(freed,['L']);self.assertFalse(parent.requests)
            self.assertFalse(nx.seen[-1][0])

    def test_only_nixl_stale_is_still_removed(self):
        nx=self.child({'old'},nixl=True);connector=self.multi([nx,self.child()])
        parent,freed=self.parent(connector,[]);parent._update_from_kv_xfer_finished(self.frame(connector))
        self.assertFalse(freed)

    def test_valid_nixl_and_mixed_foreign_completion(self):
        nx=self.child({'N','old'},nixl=True,pending={'N'},awaiting={'N'})
        connector=self.multi([nx,self.child({'L'})]);parent,freed=self.parent(connector,['N','L'])
        parent._update_from_kv_xfer_finished(self.frame(connector));self.assertEqual(set(freed),{'N','L'})

    def test_early_owned_nixl_veto_cannot_be_readded_by_same_rid_sibling(self):
        nx=self.child({'R'},nixl=True,pending={'R'})
        connector=self.multi([nx,self.child({'R'})]);frame=self.frame(connector)
        parent,freed=self.parent(connector,['R']);parent.requests['R'].is_finished=lambda:False
        parent._update_from_kv_xfer_finished(frame)
        self.assertFalse(freed);self.assertEqual(nx.connector_scheduler._sends_completed_early,{'R'})

    def test_old_nixl_and_new_lmcache_same_rid_keep_only_valid_source(self):
        nx=self.child({'R'},nixl=True);lc=self.child({'R'})
        connector=self.multi([nx,lc]);parent,freed=self.parent(connector,['R'])
        parent._update_from_kv_xfer_finished(self.frame(connector));self.assertEqual(freed,['R'])

    def test_nixl_failure_injection_is_merged_but_not_broadcast_to_sibling(self):
        nx=self.child(nixl=True);nx.connector_scheduler._scheduler_failed_recvs={'F'}
        nx.connector_scheduler._live_recv_reqs={'F'};lc=self.child()
        connector=self.multi([nx,lc]);frame=self.frame(connector);connector.update_connector_output(frame)
        self.assertEqual(frame.finished_recving,{'F'});self.assertEqual(frame.failed_recving,{'F'})
        self.assertFalse(lc.seen[-1][1]);self.assertFalse(lc.seen[-1][2])

    def test_sibling_receive_does_not_certify_nixl_prefix_state(self):
        nx=self.child(nixl=True);seen=[]
        nx.connector_scheduler.prefix_state=SimpleNamespace(completed=lambda out:seen.append(set(out.finished_recving or ())))
        connector=self.multi([nx,self.child(recvs={'L'})]);connector.update_connector_output(self.frame(connector))
        self.assertEqual(seen,[set()]);self.assertFalse(nx.connector_scheduler._recv_release_proofs)

    def test_local_cross_step_extra_save_keeps_early_child_source(self):
        nx=self.child({'R'},nixl=True,pending={'R'},awaiting={'R'});lc=self.child()
        lc.reports.append(Transfer({'R'}));connector=self.multi([nx,lc]);connector._extra_async_saves={'R':1}
        first=self.frame(connector);self.assertFalse(first.finished_sending)
        second=self.frame(connector);self.assertEqual(second.kv_connector_worker_meta.send_sources,{'R':{0,1}})
        parent,freed=self.parent(connector,['R']);parent._update_from_kv_xfer_finished(second)
        self.assertEqual(freed,['R']);connector.update_connector_output(self.frame(connector));self.assertEqual(freed,['R'])

    def test_scheduler_keeps_sources_until_global_barrier_exit(self):
        nx=self.child(nixl=True);lc=self.child();connector=self.multi([nx,lc])
        meta=self.module.MultiCompletionWorkerMetadata((None,None),{'L':{1}},{})
        connector.update_connector_output(Transfer(kv_connector_worker_meta=meta))
        self.assertEqual(connector._ilu_multi_scheduler_send_sources,{'L':{1}})
        parent,freed=self.parent(connector,['L']);parent._update_from_kv_xfer_finished(Transfer({'L'}))
        self.assertEqual(freed,['L']);self.assertFalse(connector._ilu_multi_scheduler_send_sources)

    def test_actual_aggregator_each_step_reaches_actual_parent_before_final_rank(self):
        w0=self.multi([self.child(nixl=True),self.child({'L'})])
        late=self.child();late.reports.append(Transfer({'L'}))
        w1=self.multi([self.child(nixl=True),late])
        scheduler=self.multi([self.child(nixl=True),self.child()])
        parent,freed=self.parent(scheduler,['L']);aggregator=self.Aggregator(2)
        def step():
            result=aggregator.aggregate([SimpleNamespace(kv_connector_output=self.frame(w0)),
                                         SimpleNamespace(kv_connector_output=self.frame(w1))])
            parent._update_from_kv_xfer_finished(result.kv_connector_output)
        step();self.assertFalse(freed)
        self.assertEqual(scheduler._ilu_multi_scheduler_send_sources,{'L':{1}})
        step();self.assertEqual(freed,['L'])
        self.assertFalse(scheduler._ilu_multi_scheduler_send_sources)
        step();self.assertEqual(freed,['L'])

    def test_actual_local_and_rank_barriers_retain_both_sources_then_free_once(self):
        n0=self.child({'R'},nixl=True);l0=self.child();l0.reports.append(Transfer({'R'}))
        n1=self.child(nixl=True);n1.reports.extend([Transfer(),Transfer({'R'})])
        l1=self.child({'R'})
        workers=[self.multi([n0,l0]),self.multi([n1,l1])]
        for worker in workers:worker._extra_async_saves={'R':1}
        nx=self.child(nixl=True,pending={'R'},awaiting={'R'})
        scheduler=self.multi([nx,self.child()]);parent,freed=self.parent(scheduler,['R'])
        aggregator=self.Aggregator(2)
        def step():
            result=aggregator.aggregate([SimpleNamespace(kv_connector_output=self.frame(w)) for w in workers])
            parent._update_from_kv_xfer_finished(result.kv_connector_output)
        step();self.assertFalse(freed)
        step();self.assertFalse(freed)
        self.assertEqual(scheduler._ilu_multi_scheduler_send_sources,{'R':{0,1}})
        step();self.assertEqual(freed,['R'])
        self.assertFalse(scheduler._ilu_multi_scheduler_send_sources)
        self.assertTrue(all(not w._ilu_multi_worker_send_sources for w in workers))
        step();self.assertEqual(freed,['R'])

    def test_empty_fast_path_and_single_child_bypass(self):
        children=[self.child(nixl=True),self.child()];connector=self.multi(children)
        value=self.frame(connector);self.assertIsNone(value.kv_connector_worker_meta)
        connector.update_connector_output(value);self.assertFalse(value.finished_sending)
        nx=self.child({'old'},nixl=True);single=self.multi([nx]);value=self.frame(single)
        self.assertIsNone(value.kv_connector_worker_meta);single.update_connector_output(value)
        self.assertFalse(value.finished_sending)

    def test_missing_source_is_fail_closed(self):
        connector=self.multi([self.child(nixl=True),self.child()])
        with self.assertRaisesRegex(ValueError,'Missing Multi'):connector.update_connector_output(Transfer({'L'}))

    def test_child_metadata_slot_and_parent_are_preserved(self):
        nx,lc=self.child(nixl=True),self.child();connector=self.multi([nx,lc])
        left,right=object(),object();meta=self.module.MultiCompletionWorkerMetadata((left,right))
        frame=Transfer(kv_connector_worker_meta=meta);connector.update_connector_output(frame)
        self.assertIs(nx.seen[-1][3],left);self.assertIs(lc.seen[-1][3],right)
        self.assertIs(frame.kv_connector_worker_meta,meta)

    def test_unpacked_worker_sources_cannot_be_silently_overwritten(self):
        connector=self.multi([self.child({'R'}),self.child()])
        connector.get_transfer_results(set())
        with self.assertRaisesRegex(RuntimeError,'not packed'):connector.get_transfer_results(set())
        connector.build_connector_worker_meta();connector.get_transfer_results(set())

    def test_loader_fingerprints_use_complete_dedented_fixed_method_source(self):
        registration=REPO/'vllm_iluvatar/ilu_patch/general_patches/patch_multi_completion_sources.py'
        tree=ast.parse(registration.read_text(encoding='utf-8'))
        expected=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                      and any(isinstance(target,ast.Name) and target.id=='_FINGERPRINTS' for target in node.targets))
        # inspect.getsource returns complete source lines, including the first
        # line's indentation; AST.get_source_segment does not preserve that.
        lines=self.multi_file.read_text(encoding='utf-8').splitlines()
        cls=next(node for node in ast.parse('\n'.join(lines)).body if isinstance(node,ast.ClassDef) and node.name=='MultiConnector')
        actual={node.name:hashlib.sha256(textwrap.dedent('\n'.join(lines[node.lineno-1:node.end_lineno])).strip().encode()).hexdigest()
                for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in expected}
        self.assertEqual(actual,expected)

    def test_metadata_maps_do_not_keep_a_future_or_event_alive(self):
        class Witness:pass
        event=Witness();weak=weakref.ref(event)
        connector=self.multi([self.child({'R'}),self.child()])
        frame=self.frame(connector);frame.event_for_test=event
        connector.update_connector_output(frame)
        for child in connector._connectors:child.seen.clear()
        del frame,event;gc.collect();self.assertIsNone(weak())
        self.assertFalse(connector._ilu_multi_worker_send_sources)
        self.assertFalse(connector._ilu_multi_scheduler_send_sources)


if __name__=='__main__':unittest.main(verbosity=2)

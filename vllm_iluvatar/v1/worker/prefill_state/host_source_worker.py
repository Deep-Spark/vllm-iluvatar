"""Capture complete current-request prefix states on the producer stream."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import types

import numpy as np
import torch
from vllm_iluvatar.v1.worker.gpu_worker import IluWorker
from .source_state import StatePublisher, local_paths, request_dir, replay_layout
from .cache_rows import copy_indexer_row_to_host


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value) + '\n')
    tmp.replace(path)


class HostSourceWorker(IluWorker):
    def load_model(self, *args, **kwargs):
        value = super().load_model(*args, **kwargs)
        from vllm.distributed import get_pp_group, get_tp_group
        from vllm_iluvatar.models.deepseek_v41 import model as module
        self._src_pp = get_pp_group().rank_in_group
        self._src_tp = get_tp_group().rank_in_group
        self._src_rank = self._src_pp * 2 + self._src_tp
        self._src_engine = self.vllm_config.kv_transfer_config.engine_id
        self._src_wire, self._src_nonce = local_paths(self._src_engine)
        self._src_result = self._src_wire / 'source'
        self._src_cap = self.vllm_config.scheduler_config.max_num_seqs
        self._src_stop = threading.Event()
        self._src_thread = None
        models = [m for m in self.model_runner.get_model().modules()
                  if isinstance(m, module.IluDeepseekV41Model)]
        assert len(models) == 1
        self._src_model = models[0]
        self._src_ready = False
        self._src_lock = threading.Lock()
        self._src_active = {}
        self._src_export = deque()
        self._src_free = deque()
        self._src_generations = {}
        self._src_batch = None
        self._src_counts = dict(admitted=0, full=0, partial=0, capacity=0, aborted=0)
        runner = self.model_runner
        original_add = runner.req_states.add_request
        original_remove = runner.req_states.remove_request
        original_prepare = runner.prepare_inputs
        original_execute = runner.execute_model
        original_finish = runner.finish_requests

        def add(req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens):
            result = original_add(req_id, prompt_len, all_token_ids,
                                  num_computed_tokens, max_tokens)
            if not self._src_ready:
                return result
            idx = runner.req_states.req_id_to_index[req_id]
            gen = self._src_generations.get(idx, 0) + 1
            self._src_generations[idx] = gen
            # Exact CPU input identity; no device->host read or JIT hashing.
            key = hashlib.sha256(np.asarray(all_token_ids[:prompt_len],
                                           dtype=np.int32).tobytes()).hexdigest()
            eligible = prompt_len == len(all_token_ids) and prompt_len >= 128
            hit, start, rows = replay_layout(prompt_len) if eligible else (prompt_len, 0, 0)
            with self._src_lock:
                pool = self._src_free.popleft() if eligible and self._src_free else None
                state = dict(id=req_id, idx=idx, generation=gen, N=prompt_len,
                             H=hit, S=start, token_sha256=key,
                             initial_computed=int(num_computed_tokens), run_uuid=self._src_nonce,
                             engine_id=self._src_engine,
                             blocks=[[] for _ in range(runner.block_tables.num_kv_cache_groups)],
                             pool=pool, coverage=[False]*rows, spans=[], events=[], records=[],
                             phase='ACTIVE', abort=False, fresh=False,
                             add_mono_ns=time.monotonic_ns())
                self._src_active[req_id] = state
                self._src_counts['admitted'] += 1
                if eligible and pool is None:
                    self._src_counts['capacity'] += 1
            directory = request_dir(self._src_result, req_id)
            directory.mkdir(exist_ok=True)
            save(directory/f'observed_{self._src_rank}.json', self._scalar(state))
            return result

        def remove(req_id):
            result = original_remove(req_id)
            if result is not None:
                with self._src_lock:
                    state = self._src_active.pop(req_id, None)
                    if state and state['phase'] == 'ACTIVE':
                        state['abort'] = True
                        state['phase'] = 'EXPORT_PENDING'
                        self._src_export.append(state)
            return result

        def prepare(*args, **kwargs):
            batch = original_prepare(*args, **kwargs)
            if self._src_ready:
                q = batch.query_start_loc_np[:batch.num_reqs+1]
                self._src_batch = [dict(id=rid, idx=int(batch.idx_mapping_np[j]),
                    q0=int(q[j]), q=int(q[j+1]-q[j]), c=int(batch.num_computed_tokens_np[j]),
                    N=int(batch.prefill_len_np[j]), prefill=bool(batch.is_prefilling_np[j]))
                    for j, rid in enumerate(batch.req_ids)]
            return batch

        def execute(*args, **kwargs):
            try:
                return original_execute(*args, **kwargs)
            finally:
                self._src_batch = None

        def finish(output):
            with self._src_lock:
                for rid in output.preempted_req_ids:
                    state = self._src_active.get(rid)
                    if state and state['phase'] == 'ACTIVE':
                        state['abort'] = True
            return original_finish(output)

        runner.req_states.add_request = add
        runner.req_states.remove_request = remove
        runner.prepare_inputs = prepare
        runner.execute_model = execute
        runner.finish_requests = finish
        namespace = dict(vars(module), _p21_capture=self.capture)
        exec((Path(__file__).parent/'source_model_forward.py').read_text(), namespace)
        self._src_model.forward = types.MethodType(namespace['forward'], self._src_model)
        return value

    def _scalar(self, state):
        return {k: state[k] for k in ('id', 'idx', 'generation', 'N', 'H', 'S',
                'token_sha256', 'initial_computed', 'run_uuid', 'engine_id', 'add_mono_ns')}

    def compile_or_warm_up_model(self, *args, **kwargs):
        value = super().compile_or_warm_up_model(*args, **kwargs)
        runner = self.model_runner
        old_append = runner.block_tables.append_block_ids

        def append(idx, new_block_ids, overwrite):
            rid = runner.req_states.index_to_req_id.get(idx)
            with self._src_lock:
                state = self._src_active.get(rid)
                if state and state['phase'] == 'ACTIVE' and state['generation'] == self._src_generations[idx]:
                    for g, ids in enumerate(new_block_ids):
                        bpk = runner.block_tables.blocks_per_kv_block[g]
                        expanded = [b*bpk+k for b in ids for k in range(bpk)]
                        state['blocks'][g] = expanded if overwrite else state['blocks'][g]+expanded
            return old_append(idx, new_block_ids, overwrite)

        runner.block_tables.append_block_ids = append
        static = self.vllm_config.compilation_config.static_forward_context
        self._src_specs = []
        for g, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                if '.layers.' not in name:
                    continue
                layer_id = int(name.split('.layers.')[1].split('.')[0])
                if not self._src_model.start_layer <= layer_id < self._src_model.end_layer:
                    continue
                cache = getattr(static.get(name), 'kv_cache', None)
                if not isinstance(cache, torch.Tensor) or not cache.numel():
                    continue
                kind = ('SWA' if name.endswith('.attn.swa_cache') else
                        'ring' if name.endswith('.attn.compressor.state_cache') else None)
                if kind:
                    self._src_specs.append(dict(name=name, kind=kind, layer=layer_id,
                                                group=g, cache=cache, R=int(cache.shape[1]),
                                                ratio=1 if layer_id==20 else 2))
                elif layer_id == 20 and (name.endswith('.attn') or name.endswith('.attn.indexer.k_cache')):
                    kind = 'main_tail' if name.endswith('.attn') else 'indexer_tail'
                    self._src_specs.append(dict(name=name, kind=kind, layer=20,
                                                group=g, cache=cache, R=int(cache.shape[1]), ratio=1))
        assert sum(s['kind']=='SWA' for s in self._src_specs) == self._src_model.end_layer-self._src_model.start_layer
        assert sum(s['kind']=='ring' for s in self._src_specs) == (0 if self._src_pp==3 else 1)
        before = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        host_bytes = 0
        for _ in range(self._src_cap):
            pool = {}
            for s in self._src_specs:
                if s['kind']=='SWA':
                    shape, dtype = (129, 512), torch.bfloat16
                elif s['kind']=='ring':
                    shape, dtype = (16, 1024), torch.float32
                elif s['kind']=='main_tail':
                    assert s['cache'].dtype == torch.bfloat16 and s['cache'].shape[-1] == 512
                    shape, dtype = (1, 512), torch.bfloat16
                else:
                    assert s['kind']=='indexer_tail'
                    width = 132 if s['cache'].dtype == torch.uint8 else 256
                    shape, dtype = (width,), torch.uint8
                pool[s['name']] = torch.empty(shape, dtype=dtype, pin_memory=True)
            if self._src_pp==3:
                pool.update(HC=torch.empty((129,4,5120), dtype=torch.bfloat16, pin_memory=True),
                            pre_mix=torch.empty((129,4), dtype=torch.float32, pin_memory=True),
                            topk20=torch.empty((129,512), dtype=torch.int32, pin_memory=True),
                            candidates20=torch.empty((129,2048), dtype=torch.int32, pin_memory=True))
            host_bytes += sum(t.numel()*t.element_size() for t in pool.values())
            self._src_free.append(pool)
        assert before == (torch.cuda.memory_allocated(), torch.cuda.memory_reserved()), 'source GPU owner changed'
        self._src_ready = True
        save(self._src_result/f'installed_{self._src_rank}.json', dict(rank=self._src_rank, pp=self._src_pp,
            tp=self._src_tp, active_cap=self._src_cap, pinned_host_bytes=host_bytes, new_GPU_owner_bytes=0,
            max_model_len=self.vllm_config.model_config.max_model_len,
            specs=[{k:v for k,v in s.items() if k!='cache'} for s in self._src_specs]))
        self._src_publisher = (StatePublisher(self._src_wire, self._src_engine, self._src_nonce,
            float(os.environ.get('P21_SHARD_TIMEOUT_S', '3600')),
            float(os.environ.get('P21_PACKET_TTL_S', '60')))
            if (self._src_pp, self._src_tp) == (3, 0) else None)
        self._src_thread = threading.Thread(target=self._export_loop, daemon=True, name='P21-host-export')
        self._src_thread.start()
        return value

    def capture(self, model, hc, pre_mix, positions):
        if not self._src_ready or self._src_batch is None:
            return
        for row in self._src_batch:
            if not row['prefill']:
                continue
            with self._src_lock:
                state = self._src_active.get(row['id'])
            if state is None or state['phase'] != 'ACTIVE':
                continue
            assert state['idx']==row['idx'] and state['generation']==self._src_generations[row['idx']]
            assert state['N']==row['N']
            pool = state['pool']
            a, b = max(row['c'], state['S']), min(row['c']+row['q'], state['N'])
            if pool is not None and not state['abort'] and a < b and self._src_pp==3:
                src = slice(row['q0']+a-row['c'], row['q0']+b-row['c'])
                dst = slice(a-state['S'], b-state['S'])
                assert not any(state['coverage'][dst]), 'duplicate HC span'
                for name, source in (('HC',hc), ('pre_mix',pre_mix),
                                     ('topk20',model.topk_indices_buffer),
                                     ('candidates20',model.candidate_idxs_buffer)):
                    pool[name][dst].copy_(source[src], non_blocking=True)
                state['coverage'][dst] = [True]*(b-a)
            state['spans'].append(dict(computed=row['c'], query=row['q']))
            event = torch.cuda.Event();event.record()
            state['events'].append(event)
            if row['c']+row['q'] == state['N']:
                state['fresh'] = (pool is not None and not state['abort'] and
                                  min(span['computed'] for span in state['spans']) <= state['S'] and
                                  (self._src_pp!=3 or all(state['coverage'])))
                if state['fresh']:
                    state['records'] = self._copy_caches(state)
                    event = torch.cuda.Event();event.record()
                    state['events'].append(event)
                state['phase'] = 'EXPORT_PENDING'
                with self._src_lock:
                    self._src_export.append(state)

    def _copy_caches(self, state):
        records=[]
        for spec in self._src_specs:
            cache=spec['cache'];host=state['pool'][spec['name']]
            blocks=state['blocks'][spec['group']];segments=[]
            if spec['kind'] in ('main_tail', 'indexer_tail'):
                if state['N'] == state['H']:
                    continue
                start, stop = state['H'], state['N']
                logical, row = divmod(start, spec['R'])
                assert logical < len(blocks) and 0 < int(blocks[logical]) < cache.shape[0]
                block = int(blocks[logical])
                if spec['kind'] == 'main_tail':
                    host[0].copy_(cache[block, row], non_blocking=True)
                else:
                    copy_indexer_row_to_host(host, cache, block, row)
                segments.append(dict(logical_rows=[start,stop], physical_block=block,
                                     physical_rows=[row,row+1]))
            elif spec['kind']=='ring':
                assert len(blocks)==1 and 0<int(blocks[0])<cache.shape[0]
                host.copy_(cache[int(blocks[0])],non_blocking=True)
                start,stop=state['N']-16,state['N']
            else:
                start,stop=state['S'],state['N'];p=start
                while p<stop:
                    logical,lo=divmod(p,64);count=min(stop-p,64-lo)
                    assert logical<len(blocks) and 0<int(blocks[logical])<cache.shape[0]
                    block=int(blocks[logical]);off=p-start
                    host[off:off+count].copy_(cache[block,lo:lo+count],non_blocking=True)
                    segments.append(dict(logical_rows=[p,p+count],physical_block=block,
                                         physical_rows=[lo,lo+count]))
                    p+=count
            records.append(dict(name=spec['name'],kind=spec['kind'],layer=spec['layer'],
                group=spec['group'],dtype=str(host.dtype),
                shape=([state['N']-state['S'],512] if spec['kind']=='SWA' else list(host.shape)),
                absolute_interval=[start,stop],segments=segments))
        return records

    def _export_loop(self):
        torch.cuda.set_device(self.device)
        report_time = 0
        while not self._src_stop.is_set():
            with self._src_lock:
                state = self._src_export[0] if self._src_export else None
            if state is not None and all(event.query() for event in state['events']):
                try:
                    meta = self._scalar(state)
                    meta.update(rank=self._src_rank, pp=self._src_pp, tp=self._src_tp,
                                fresh=state['fresh'], spans=state['spans'], abort=state['abort'],
                                export_mono_ns=time.monotonic_ns(), records=[])
                    directory = request_dir(self._src_result, state['id'])
                    directory.mkdir(exist_ok=True)
                    if state['fresh']:
                        # CPU-only serialization after all writer-stream D2H events.
                        pieces = []
                        offset = 0
                        records = list(state['records'])
                        if self._src_pp==3:
                            records += [dict(name=k, kind='boundary21', dtype=str(state['pool'][k].dtype),
                                             shape=[state['N']-state['S'],*state['pool'][k].shape[1:]])
                                        for k in ('HC','pre_mix','topk20','candidates20')]
                        for record in records:
                            host = state['pool'][record['name']]
                            if tuple(host.shape) != tuple(record['shape']):
                                host = host[:record['shape'][0]]
                            raw = host.contiguous().view(torch.uint8).numpy().tobytes()
                            if host.dtype==torch.bfloat16:
                                # CPU bit-level finite check avoids a large
                                # FP32 conversion/CPU-thread-pool in exporters.
                                words=np.frombuffer(raw,dtype=np.uint16)
                                assert bool(np.all((words & 0x7f80)!=0x7f80))
                            elif host.dtype==torch.float32:
                                assert bool(np.isfinite(np.frombuffer(raw,dtype=np.float32)).all())
                            meta['records'].append({**record, 'offset_bytes':offset,
                                                    'byte_count':len(raw)})
                            pieces.append(raw);offset += len(raw)
                        raw = b''.join(pieces)
                        (directory/f'rank{self._src_rank}.bin').write_bytes(raw)
                        meta.update(byte_count=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                        self._src_counts['full'] += 1
                    elif state['abort']:
                        self._src_counts['aborted'] += 1
                    else:
                        self._src_counts['partial'] += 1
                    save(directory/f'auto_{self._src_rank}.json',meta)
                except Exception as error:
                    save(self._src_result/f'error_{self._src_rank}.json',dict(error=repr(error),request_id=state['id']))
                finally:
                    with self._src_lock:
                        assert self._src_export.popleft() is state
                        state['phase']='EXPORTED'
                        if state['pool'] is not None:
                            self._src_free.append(state.pop('pool'))
                        state['events']=[]
                        state['blocks']=[]
            if time.monotonic()-report_time>=1:
                save(self._src_result/f'counters_{self._src_rank}.json',self._src_counts)
                report_time = time.monotonic()
            if self._src_publisher is not None:
                self._src_publisher.poll()
            self._src_stop.wait(.002)

    def shutdown(self):
        self._src_ready = False
        if hasattr(self, '_src_stop'):
            self._src_stop.set()
        if getattr(self, '_src_thread', None) is not None:
            self._src_thread.join()
        super().shutdown()

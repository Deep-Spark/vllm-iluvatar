"""Import current producer states and execute the remaining mixed batch."""
import copy
import hashlib
import json
import math
import threading
import time

import numpy as np
import torch
from vllm_iluvatar.v1.worker.gpu_worker import IluWorker
from .source_state import DTYPES, fetch_state, local_paths, replay_layout
from .decode_prefix import PrefixBuffers
from .cache_rows import indexer_row_views, restore_indexer_row

def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value)+'\n');tmp.replace(path)

class LiveDWorker(IluWorker):
    def load_model(self,*args,**kwargs):
        result=super().load_model(*args,**kwargs)
        from vllm.distributed import get_ep_group,get_dp_group,get_tp_group
        from vllm_iluvatar.models.deepseek_v41 import model as module
        from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.connector import OptNixlConnector
        ep=get_ep_group()
        self._p21_rank=ep.rank_in_group
        self._p21_dp=get_dp_group().rank_in_group
        self._p21_tp=get_tp_group().rank_in_group
        self._p21_engine=self.vllm_config.kv_transfer_config.engine_id
        self._p21_wire,self._p21_nonce=local_paths(self._p21_engine)
        self._p21_result=self._p21_wire/'decode'
        self._p21_lock=threading.Lock()
        self._p21_stop=threading.Event()
        self._p21_thread=None
        self._p21_ready=False
        self._p21_states={}
        self._p21_generation={}
        self._p21_packets={}
        self._p21_host_inflight=[]
        self._p21_unavailable=set()
        self._p21_remote_runs={}
        self._p21_recv={}
        self._p21_recv_metadata={}
        self._p21_full_ready={}
        self._p21_reasons={}
        self._p21_batch=None
        self._p21_builder_scope=False
        self._p21_dummy=False
        self._p21_eager_warmed=False
        self._p21_raw_tables=self._p21_raw_slots=None
        self._p21_counters={'NONE':0,'GO':0,'fallback':0,'skipped_tail_rows':0,
                            'tail128_requests':0,'tail129_requests':0,
                            'ready_packets':0,'source_unavailable':0}
        models=[m for m in self.model_runner.get_model().modules() if isinstance(m,module.IluDeepseekV41Model)]
        assert len(models)==1
        self._p21_target=models[0]
        assert (self._p21_target.start_layer,self._p21_target.end_layer)==(0,40)
        self._p21_max_reqs=self.vllm_config.scheduler_config.max_num_seqs
        spec=self.vllm_config.speculative_config
        self._p21_query_len=1+(int(getattr(spec,'num_speculative_tokens',0) or 0) if spec else 0)
        self._p21_max_decode=self._p21_max_reqs*self._p21_query_len
        tp_size=get_tp_group().world_size
        max_tokens=(self._p21_max_decode+tp_size-1)//tp_size*tp_size
        before_alloc=torch.cuda.memory_allocated(self.device),torch.cuda.memory_reserved(self.device)
        self._p21_prefix=PrefixBuffers(max_tokens,self._p21_target.config.hidden_size,
                                      self._p21_target.hc_mult,self.device)
        self._p21_prefix_memory=dict(owner_bytes=self._p21_prefix.owner.numel(),
            allocated_bytes=torch.cuda.memory_allocated(self.device)-before_alloc[0],
            reserved_bytes=torch.cuda.memory_reserved(self.device)-before_alloc[1])
        original_forward=self._p21_target.forward
        def forward(*args,**kwargs):
            if not self._p21_ready:return original_forward(*args,**kwargs)
            return self._p21_forward(original_forward,*args,**kwargs)
        self._p21_target.forward=forward
        runner=self.model_runner
        oldadd,oldremove=runner.req_states.add_request,runner.req_states.remove_request
        oldprepare=runner.prepare_inputs
        oldattn=runner.prepare_attn
        oldexecute=runner.execute_model
        def add(req_id,prompt_len,all_token_ids,num_computed_tokens,max_tokens):
            rid=req_id
            value=oldadd(rid,prompt_len,all_token_ids,num_computed_tokens,max_tokens)
            idx=runner.req_states.req_id_to_index[rid]
            gen=self._p21_generation.get(idx,0)+1;self._p21_generation[idx]=gen
            with self._p21_lock:
                self._p21_states[rid]={'id':rid,'idx':idx,'generation':gen,'N':prompt_len,
                    'token_sha256':hashlib.sha256(np.asarray(all_token_ids[:prompt_len],dtype=np.int32).tobytes()).hexdigest(),
                    'blocks':[[] for _ in range(runner.block_tables.num_kv_cache_groups)]}
            return value
        def remove(rid):
            value=oldremove(rid)
            if value is None:
                # Admission first removes an unknown old id; retain the already received KV.
                return value
            with self._p21_lock:
                self._p21_states.pop(rid,None)
                packet=self._p21_packets.pop(rid,None)
                self._p21_recv.pop(rid,None)
                self._p21_recv_metadata.pop(rid,None)
                self._p21_full_ready.pop(rid,None)
                self._p21_unavailable.discard(rid)
            # Imported pinned hosts stay in _p21_host_inflight until the current
            # stream's copy event completes, including cancellation/slot reuse.
            del packet
            return value
        def prepare(*args,**kwargs):
            batch=oldprepare(*args,**kwargs);self._p21_batch=batch;return batch
        def attn(batch):
            tables,slots=oldattn(batch)
            self._p21_raw_tables=tables;self._p21_raw_slots=slots
            return tables,slots
        def execute(*args,**kwargs):
            self._p21_dummy=bool(kwargs.get('dummy_run',False))
            try:return oldexecute(*args,**kwargs)
            finally:
                self._p21_dummy=False
                self._p21_batch=None
                self._p21_raw_tables=self._p21_raw_slots=None
        oldmetadata=runner.model_state.prepare_attn
        def metadata(input_batch,cudagraph_mode,block_tables,slot_mappings,*args,**kwargs):
            if self._p21_ready and not self._p21_builder_scope:
                # Includes actual idle-DP dummy runs, which bypass prepare_inputs.
                self._p21_batch=input_batch
                self._p21_raw_tables=block_tables
                self._p21_raw_slots=slot_mappings
            return oldmetadata(input_batch,cudagraph_mode,block_tables,slot_mappings,*args,**kwargs)
        runner.model_state.prepare_attn=metadata
        runner.req_states.add_request=add;runner.req_states.remove_request=remove
        runner.prepare_inputs=prepare;runner.prepare_attn=attn;runner.execute_model=execute
        oldbind=OptNixlConnector.bind_connector_metadata
        oldresults=OptNixlConnector.get_transfer_results
        def bind(connector,metadata):
            if self._p21_ready:
                new_ready=getattr(metadata,'source_full_ready_receipts',{})
                self._p21_full_ready.update(new_ready)
                if new_ready and not (self._p21_result/f'full_ready_{self._p21_rank}.json').exists():
                    save(self._p21_result/f'full_ready_{self._p21_rank}.json',new_ready)
                for rid,meta in getattr(metadata,'reqs_to_recv',{}).items():
                    self._p21_recv_metadata[rid]={'prompt_token_len':meta.prompt_token_len,
                        'local_block_ids':copy.deepcopy(meta.local_block_ids),
                        'engine_id':meta.engine_id,
                        'source_endpoint':getattr(metadata,'source_state_endpoints',{}).get(rid)}
            return oldbind(connector,metadata)
        def results(connector,finished_req_ids):
            value=oldresults(connector,finished_req_ids)
            for rid in value.failed_recving or set():
                self._p21_recv.pop(rid,None);self._p21_full_ready.pop(rid,None)
            for rid in (value.finished_recving or set())-(value.failed_recving or set()):
                event=torch.cuda.Event();event.record()
                self._p21_recv[rid]={'event':event,
                    'metadata':self._p21_recv_metadata.get(rid),
                    'post_layout_original_completion':True}
            return value
        OptNixlConnector.bind_connector_metadata=bind
        OptNixlConnector.get_transfer_results=results
        return result

    def compile_or_warm_up_model(self,*args,**kwargs):
        value=super().compile_or_warm_up_model(*args,**kwargs)
        runner=self.model_runner
        original=runner.block_tables.append_block_ids
        def append(idx,new_block_ids,overwrite):
            with self._p21_lock:
                rid=runner.req_states.index_to_req_id.get(idx)
                state=self._p21_states.get(rid)
                if state and state['idx']==idx and self._p21_generation[idx]==state['generation']:
                    for i,ids in enumerate(new_block_ids):
                        bpk=runner.block_tables.blocks_per_kv_block[i]
                        expanded=[b*bpk+k for b in ids for k in range(bpk)]
                        state['blocks'][i]=expanded if overwrite else state['blocks'][i]+expanded
            return original(idx,new_block_ids,overwrite)
        runner.block_tables.append_block_ids=append
        self._p21_cache_sources=[]
        static=self.vllm_config.compilation_config.static_forward_context
        for g,group in enumerate(runner.kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                cache=getattr(static.get(name),'kv_cache',None)
                if not isinstance(cache,torch.Tensor) or not cache.numel() or '.layers.' not in name:continue
                layer=int(name.split('.layers.')[1].split('.')[0])
                kind='SWA' if name.endswith('.attn.swa_cache') else 'ring' if name.endswith('.attn.compressor.state_cache') else None
                if kind is None and layer == 20:
                    kind = ('main_tail' if name.endswith('.attn') else
                            'indexer_tail' if name.endswith('.attn.indexer.k_cache') else None)
                if kind and layer<=20:self._p21_cache_sources.append({'name':name,'group':g,
                                                        'layer':layer,'kind':kind,'cache':cache})
        assert len(self._p21_cache_sources)==26
        self._p21_ready=True
        save(self._p21_result/f'installed_{self._p21_rank}.json',{'rank':self._p21_rank,'dp':self._p21_dp,
             'tp':self._p21_tp,'actual_aux':list(self._p21_target.aux_hidden_state_layers),
             'new_graphs':0,'prefix_memory':self._p21_prefix_memory,'GPU_packet_bytes':0,
             'max_num_seqs':self._p21_max_reqs,
             'max_model_len':self.vllm_config.model_config.max_model_len})
        def receive():
            try:self._p21_receiver()
            except Exception as error:
                save(self._p21_result/f'error_receiver_{self._p21_rank}.json',{'error':repr(error)})
                raise
        self._p21_thread=threading.Thread(target=receive,daemon=True,name='P21-receiver')
        self._p21_thread.start()
        return value

    def _p21_receiver(self):
        import zmq
        torch.cuda.set_device(self.device)
        report_time=0;sockets={}
        context=zmq.Context()
        try:
            while not self._p21_stop.is_set():
                with self._p21_lock:
                    self._p21_host_inflight[:]=[hold for hold in self._p21_host_inflight
                                               if hold['event'] is None or not hold['event'].query()]
                    pending=[{'id':rid,'idx':s['idx'],'generation':s['generation'],'N':s['N'],
                              'token_sha256':s['token_sha256'],
                              'endpoint':self._p21_recv_metadata.get(rid,{}).get('source_endpoint')}
                             for rid,s in self._p21_states.items()
                             if rid not in self._p21_packets and rid not in self._p21_unavailable
                             and not s.get('imported') and s['N']>=128]
                for state in pending:
                    if self._p21_stop.is_set():break
                    endpoint=state['endpoint'];rid=state['id']
                    if endpoint is None:continue
                    key=(endpoint['engine_id'],endpoint['addr'])
                    try:
                        if key not in sockets:
                            sock=context.socket(zmq.REQ)
                            sock.setsockopt(zmq.LINGER,0)
                            sock.setsockopt(zmq.SNDTIMEO,500)
                            sock.setsockopt(zmq.RCVTIMEO,500)
                            sock.connect('tcp://'+endpoint['addr'])
                            sockets[key]=sock
                        reply,raw=fetch_state(sockets[key],key[0],rid,self._p21_tp,
                                              state['N'],state['token_sha256'])
                        if reply['status']!='READY':
                            if reply['status']!='PENDING':
                                with self._p21_lock:self._p21_unavailable.add(rid)
                                self._p21_counters['source_unavailable']+=1
                            continue
                        metadata=reply['metadata'];source=reply['source']
                        previous=self._p21_remote_runs.setdefault(key[0],metadata['run_uuid'])
                        if previous!=metadata['run_uuid']:
                            raise ValueError('Producer service lifetime changed')
                        host=torch.frombuffer(bytearray(raw),dtype=torch.uint8).pin_memory()
                        receipt={'metadata':metadata,'source':source,'host':host,
                                 'generation':state['generation'],'idx':state['idx'],
                                 'local_nonce':self._p21_nonce,'remote_engine_id':key[0]}
                        with self._p21_lock:
                            current=self._p21_states.get(rid)
                            if (current and current['generation']==state['generation']
                                    and self._p21_recv_metadata.get(rid,{}).get('source_endpoint')==endpoint):
                                self._p21_packets[rid]=receipt
                                self._p21_counters['ready_packets']+=1
                    except zmq.ZMQError:
                        sock=sockets.pop(key,None)
                        if sock is not None:sock.close()
                    except (ValueError,KeyError,TypeError,AttributeError) as error:
                        with self._p21_lock:self._p21_unavailable.add(rid)
                        self._p21_counters['source_unavailable']+=1
                        save(self._p21_result/f'invalid_source_{self._p21_rank}.json',
                             dict(request_id=rid,error=repr(error)))
                if time.monotonic()-report_time>=1:
                    save(self._p21_result/f'counters_{self._p21_rank}.json',self._p21_counters)
                    save(self._p21_result/f'reasons_{self._p21_rank}.json',self._p21_reasons)
                    report_time=time.monotonic()
                self._p21_stop.wait(.001)
        finally:
            for sock in sockets.values():sock.close()
            context.term()

    def shutdown(self):
        self._p21_ready=False
        if hasattr(self,'_p21_stop'):self._p21_stop.set()
        if getattr(self,'_p21_thread',None) is not None:self._p21_thread.join()
        super().shutdown()

    def _p21_forward(self,original,input_ids,positions,intermediate_tensors,
                     inputs_embeds=None,lookback_token_ids=None):
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import get_forward_context,DPMetadata,override_forward_context
        from vllm.distributed import get_ep_group,get_tp_group
        from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
        from vllm_iluvatar.models.deepseek_v41 import mhc
        ctx=get_forward_context();batch=self._p21_batch
        if ctx.cudagraph_runtime_mode!=CUDAGraphMode.NONE or batch is None:
            return original(input_ids,positions,intermediate_tensors,inputs_embeds,lookback_token_ids)
        self._p21_counters['NONE']+=1
        q=batch.query_start_loc_np[:batch.num_reqs+1]
        prefill=np.asarray(batch.is_prefilling_np[:batch.num_reqs],dtype=bool)
        dummy=self._p21_dummy or not any(rid in self._p21_states for rid in batch.req_ids)
        ndecode=0 if dummy else int((~prefill).sum())
        decode=int(q[ndecode]) if ndecode else 0
        tail_indices=[] if dummy else list(range(ndecode,batch.num_reqs))
        runner=self.model_runner;model=self._p21_target
        tp_size=get_tp_group().world_size
        capacity=min(self._p21_prefix.max_tokens,32*tp_size)
        adaptive=bool(getattr(self.vllm_config.speculative_config,'enable_adaptive_verification',False))
        tails=[];eligible=(ndecode<=self._p21_max_reqs and decode<=capacity and isinstance(ctx.attn_metadata,dict) and
            np.all(~prefill[:ndecode]) and (dummy or np.all(prefill[ndecode:])) and
            self._p21_raw_tables is not None and self._p21_raw_slots is not None and not adaptive and
            not torch.cuda.is_current_stream_capturing())
        # Bounded source wait happens only on the selected local tail owner.
        # Other ranks wait at the same fixed CPU gate; P never waits for D.
        for i in tail_indices:
            rid=batch.req_ids[i]
            if self._p21_recv_metadata.get(rid,{}).get('source_endpoint') is not None:
                deadline=time.monotonic()+.050
                while time.monotonic()<deadline:
                    with self._p21_lock:
                        pending=self._p21_packets.get(rid)
                        ready=pending is not None or rid in self._p21_unavailable
                    if ready:break
                    time.sleep(.0005)
        diagnosis=[]
        with self._p21_lock:
            for i in tail_indices:
                rid=batch.req_ids[i];state=self._p21_states.get(rid)
                packet=self._p21_packets.get(rid);recv=self._p21_recv.get(rid)
                full_ready=self._p21_full_ready.get(rid)
                N=int(batch.prefill_len_np[i]);c=int(batch.num_computed_tokens_np[i])
                H, S, rows = replay_layout(N) if N >= 128 else (N, 0, 0)
                good=state is not None and packet is not None and recv is not None and full_ready is not None
                detail={'id':rid,'N':N,'c':c,'q':int(q[i+1]-q[i]),'state':state is not None,
                        'packet':packet is not None,'recv':recv is not None,'full_ready':full_ready is not None,
                        'packet_published_ns':packet['metadata']['published_mono_ns'] if packet else None,
                        'now_ns':time.monotonic_ns()}
                if good:
                    meta=packet['metadata'];receipt=recv['metadata']
                    detail['initial_conditions']={
                        'source_lengths':N==state['N']==meta['N'] and meta['H']==H and meta['S']==S,
                        'tail_computed':c==S,
                        'tail_query':int(q[i+1]-q[i])==rows,
                        'request_slot':state['idx']==int(batch.idx_mapping_np[i])==packet['idx'],
                        'generation':state['generation']==packet['generation']==self._p21_generation[state['idx']],
                        'CPU_state_ready':packet['local_nonce']==self._p21_nonce,
                        'original_receive_metadata':receipt is not None}
                    good=(N>=128 and N==state['N']==meta['N'] and meta['H']==H and meta['S']==S and
                          c==S and int(q[i+1]-q[i])==rows and
                          state['idx']==int(batch.idx_mapping_np[i])==packet['idx'] and
                          state['generation']==packet['generation']==self._p21_generation[state['idx']] and
                          packet['local_nonce']==self._p21_nonce and receipt is not None and
                          recv['post_layout_original_completion'] and
                          full_ready['request_id']==rid and full_ready['run_uuid']==self._p21_nonce and
                          full_ready['N']==N and full_ready['H']==full_ready['fullkv_valid_before_replay']==H and
                          full_ready['S']==S and full_ready['query_rows']==rows and
                          full_ready['original_receive_success'] and full_ready['external']>0 and
                          full_ready['alloc_serial']>0 and
                          full_ready['token_sha256']==meta['token_sha256']==state['token_sha256'] and
                          full_ready['remote_engine_id']==receipt['engine_id']==packet['remote_engine_id']==meta['engine_id'])
                    # Full current scheduler pages include the cached local prefix.
                    # Compact receive IDs retain their absolute logical positions.
                    if good:
                        coverage=[]
                        groups={int(g):ids for g,ids in full_ready['groups'].items()}
                        mappings={int(g):ids for g,ids in full_ready['received_positions'].items()}
                        specs={int(g):value for g,value in full_ready['group_specs'].items()}
                        good &= len(groups)==8 and set(groups)==set(mappings)==set(specs)
                        for g,ids in groups.items():
                            group=self.model_runner.kv_cache_config.kv_cache_groups[g]
                            geometry=dict(block_size=group.kv_cache_spec.block_size,
                                          layer_names=list(group.layer_names),dtype=str(group.kv_cache_spec.dtype))
                            expected=math.ceil(N/self.model_runner.block_tables.block_sizes[g])
                            bpk=self.model_runner.block_tables.blocks_per_kv_block[g]
                            expanded=[b*bpk+k for b in ids[:expected] for k in range(bpk)]
                            full_match=(len(ids)>=expected and all(int(b)>0 for b in ids[:expected]) and
                                        state['blocks'][g][:len(expanded)]==expanded and specs[g]==geometry)
                            compact=list(receipt['local_block_ids'][g]);receive_positions=mappings[g]
                            map_match=(len(compact)==len(receive_positions) and
                                       len({rec['logical'] for rec in receive_positions})==len(receive_positions))
                            for b,rec in zip(compact,receive_positions):
                                j=rec['logical']
                                map_match &= (0<=j<len(ids) and ids[j]==b==rec['block_id'] and
                                    state['blocks'][g][j*bpk:(j+1)*bpk]==[b*bpk+k for k in range(bpk)])
                            good &= full_match and map_match
                            coverage.append({'g':g,'expected':expected,'received':len(compact),
                                             'IDs_match':full_match,'compact_position_match':map_match,
                                             'first_received_logical':receive_positions[0]['logical'] if receive_positions else None})
                        detail['coverage']=coverage
                        good &= receipt['prompt_token_len'] in (-1,N)
                    if good:tails.append((i,state,packet,recv))
                eligible &= good
                detail['good']=bool(good);diagnosis.append(detail)
                reason=('eligible' if good else 'no_state' if state is None else 'no_packet' if packet is None
                        else 'no_original_recv' if recv is None else 'no_full_ready' if full_ready is None
                        else 'identity_shape_ready_or_current_pages')
                self._p21_reasons[reason]=self._p21_reasons.get(reason,0)+1
            # Validate every live SWA/ring destination before any common GO/cache write.
            for _,state,packet,_ in tails:
                for source in self._p21_cache_sources:
                    blocks=state['blocks'][source['group']];cache=source['cache']
                    if source['kind']=='SWA':
                        lo=packet['metadata']['S']//64;hi=(state['N']-1)//64
                        eligible &= len(blocks)>hi and all(0<int(blocks[j])<cache.shape[0]
                                     for j in range(lo,min(hi+1,len(blocks))))
                    elif source['kind']=='ring':
                        eligible &= len(blocks)==1 and 0<int(blocks[0])<cache.shape[0]
                    elif state['N'] % 2:
                        logical,row=divmod(state['N']-1,cache.shape[1])
                        valid=len(blocks)>logical and 0<int(blocks[logical])<cache.shape[0]
                        eligible &= valid
                        if valid and source['kind']=='indexer_tail':
                            field=next(f for f in packet['metadata']['fields'] if f['kind']=='indexer_tail')
                            eligible &= sum(v.numel() for v in indexer_row_views(cache,int(blocks[logical]),row))==field['byte_count']
                        elif valid:
                            eligible &= cache.dtype==torch.bfloat16 and cache.shape[-1]==512
        free,_=torch.cuda.mem_get_info()
        eligible &= free/2**20 >= (1100 if self._p21_eager_warmed else 1900)
        decision=torch.tensor([int(eligible),decode,len(tails),capacity,ndecode],dtype=torch.int64)
        votes=[torch.empty_like(decision) for _ in range(16)]
        torch.distributed.all_gather(votes,decision,group=get_ep_group().cpu_group)
        prefix_tokens=(max(int(v[1]) for v in votes)+tp_size-1)//tp_size*tp_size
        go=(all(int(v[0])==1 for v in votes) and any(int(v[2]) for v in votes)
            and prefix_tokens<=min(int(v[3]) for v in votes))
        if self._p21_counters['NONE']<=12:
            save(self._p21_result/f"gate_{self._p21_rank}_{self._p21_counters['NONE']}.json",
                 {'rank':self._p21_rank,'dummy':dummy,'ndecode':ndecode,'decode':decode,
                  'eligible':bool(eligible),'go':go,'details':diagnosis,
                  'votes':[v.tolist() for v in votes],'prefix_tokens':prefix_tokens,'free_MiB':free/2**20,
                  'ctx_metadata_type':type(ctx.attn_metadata).__name__})
        if not go:
            if eligible:self._p21_reasons['peer_veto']=self._p21_reasons.get('peer_veto',0)+1
            self._p21_counters['fallback']+=1
            return original(input_ids,positions,intermediate_tensors,inputs_embeds,lookback_token_ids)
        h=(inputs_embeds if inputs_embeds is not None else model.embed_input_ids(input_ids))
        h=h.unsqueeze(-2).repeat(1,model.hc_mult,1)
        pre=h.new_zeros((h.shape[0],4),dtype=torch.float32);pre[:,0]=1
        model._prepare_engram(input_ids,positions,lookback_token_ids)
        # Import only current, still-live destination pages. All CPU views are generation checked.
        for i,state,packet,recv in tails:
            # Retain before the first async copy, including partial-copy errors.
            hold={'event':None,'packet':packet}
            with self._p21_lock:self._p21_host_inflight.append(hold)
            torch.cuda.current_stream().wait_event(recv['event'])
            fields={f['name']:f for f in packet['metadata']['fields']}
            def view(field):
                dtype={'BF16':torch.bfloat16,'FP32':torch.float32,'I32':torch.int32,'U8':torch.uint8}[DTYPES[field['dtype']][0]]
                off=field['offset_bytes'];end=off+field['byte_count']
                return packet['host'][off:end].view(dtype).reshape(field['shape'])
            for source in self._p21_cache_sources:
                if source['kind'] in ('main_tail','indexer_tail') and state['N']%2==0:
                    continue
                field=next(f for f in fields.values() if f.get('kind')==source['kind']
                           and f.get('layer')==source['layer'])
                data=view(field);cache=source['cache'];blocks=state['blocks'][source['group']]
                if source['kind']=='SWA':
                    p=packet['metadata']['S'];off=0
                    while p<state['N']:
                        logical,lo=divmod(p,64);count=min(state['N']-p,64-lo)
                        block=int(blocks[logical]);assert 0<block<cache.shape[0]
                        cache[block,lo:lo+count].copy_(data[off:off+count],non_blocking=True)
                        p+=count;off+=count
                elif source['kind']=='ring':
                    assert len(blocks)==1 and 0<int(blocks[0])<cache.shape[0]
                    cache[int(blocks[0])].copy_(data,non_blocking=True)
                else:
                    logical,row=divmod(state['N']-1,cache.shape[1])
                    block=int(blocks[logical])
                    if source['kind']=='main_tail':
                        cache[block,row].copy_(data[0],non_blocking=True)
                    else:
                        restore_indexer_row(cache,block,row,data)
            region=slice(int(q[i]),int(q[i+1]))
            h[region].copy_(view(fields['HC']),non_blocking=True)
            pre[region].copy_(view(fields['pre_mix']),non_blocking=True)
            model.topk_indices_buffer[region].copy_(view(fields['topk20']),non_blocking=True)
            model.candidate_idxs_buffer[region].copy_(view(fields['candidates20']),non_blocking=True)
            consumed=torch.cuda.Event();consumed.record()
            with self._p21_lock:
                hold['event']=consumed
                state['imported']=True
                self._p21_packets.pop(state['id'],None)
        original_metadata=ctx.attn_metadata
        saved_engram=[(layer.engram,layer.engram._rows,getattr(layer.engram,'_hash_ids',None))
                      for layer in model.layers[:21] if layer.engram is not None]
        attention_methods=[];engram_methods=[]
        residual=post=res=None
        if any(int(v[1]) for v in votes):
            try:
                lease=self._p21_prefix.for_tokens(prefix_tokens)
                ph,pp=lease.hidden,lease.pre_mix
                ph.zero_();ph[:decode].copy_(h[:decode])
                pp.zero_();pp[:,0]=1;pp[:decode].copy_(pre[:decode])
                pos,ids=lease.positions,lease.input_ids
                pos.zero_();pos[:decode].copy_(positions[:decode])
                ids.zero_();ids[:decode].copy_(input_ids[:decode])
                mask=lease.is_padding;mask.fill_(True);mask[:decode].fill_(False)
                if decode:
                    sub=copy.copy(batch)
                    sub.req_ids=batch.req_ids[:ndecode]
                    sub.num_reqs=sub.num_reqs_after_padding=ndecode
                    sub.num_tokens=sub.num_tokens_after_padding=decode
                    for field in ('idx_mapping_np','num_scheduled_tokens','num_computed_tokens_np',
                                  'prefill_len_np','num_computed_prefill_tokens_np','is_prefilling_np'):
                        setattr(sub,field,getattr(batch,field)[:ndecode].copy())
                    sub.idx_mapping=batch.idx_mapping[:ndecode]
                    sub.seq_lens=batch.seq_lens[:ndecode]
                    sub.seq_lens_cpu_upper_bound=batch.seq_lens_cpu_upper_bound[:ndecode]
                    sub.query_start_loc_np=q[:ndecode+1].copy()
                    sub.query_start_loc=batch.query_start_loc[:ndecode+1]
                    sub.input_ids=input_ids[:decode]
                    sub.positions=positions[:decode]
                    sub.is_padding=mask[:decode]
                    sub.max_query_len=int(sub.num_scheduled_tokens.max());sub.has_prefill=False
                    if sub.prompt_lens is not None:sub.prompt_lens=batch.prompt_lens[:ndecode]
                    subtables=tuple(t[:ndecode] for t in self._p21_raw_tables)
                    subslots=self._p21_raw_slots[:,:decode]
                    sub.fast_prefill=None
                    self._p21_builder_scope=True
                    try:
                        md=runner.model_state.prepare_attn(sub,CUDAGraphMode.NONE,subtables,subslots,
                              runner.attn_groups,runner.kv_cache_config,for_capture=False)
                    finally:self._p21_builder_scope=False
                else:
                    # No local decode attention or cache writes. All idle ranks
                    # still enter exactly the same masked prefix MoE collectives.
                    md=ctx.attn_metadata;subslots=self._p21_raw_slots
                prefixctx=copy.copy(ctx);prefixctx.additional_kwargs=dict(ctx.additional_kwargs)
                prefixctx.attn_metadata=md
                prefixctx.slot_mapping=build_slot_mappings_by_layer(subslots,runner.kv_cache_config)
                prefixctx.dp_metadata=DPMetadata(torch.full((8,),prefix_tokens,dtype=torch.int32))
                prefixctx.dp_metadata.all_ranks_decode_only=True
                prefixctx.all_moe_layers=None;prefixctx.moe_layer_index=0;prefixctx.ubatch_slices=None
                prefixctx.is_padding=mask
                if hasattr(prefixctx,'_ilu_gathered_router_padding'):del prefixctx._ilu_gathered_router_padding
                for engram,rows,hashes in saved_engram:
                    if not decode:engram._rows=None;continue
                    engram._rows=rows[:decode];engram._hash_ids=hashes[:decode]
                    method=engram.forward;engram_methods.append((engram,method))
                    def real_engram(x,hash_ids,token_mask=None,method=method):
                        result=method(x[:decode],hash_ids[:decode])
                        x[:decode].copy_(result)
                        return x
                    engram.forward=real_engram
                for layer in model.layers[:21]:
                    method=layer.attn.forward;attention_methods.append((layer.attn,method))
                    def attention(p,x,scaling=None,method=method):
                        if not decode:return x.zero_()
                        output=method(p[:decode],x[:decode],scaling)
                        x[:decode].copy_(output);x[decode:].zero_()
                        return x
                    layer.attn.forward=attention
                with override_forward_context(prefixctx):
                    for layer in model.layers[:21]:
                        ph,residual,post,res,pp=layer(ph,pos,ids,pp,post,res,residual)
                    ph=mhc.hc_post(ph,residual,post,res)
                h[:decode].copy_(ph[:decode]);pre[:decode].copy_(pp[:decode])
            finally:
                for attn,method in attention_methods:attn.forward=method
                for engram,method in engram_methods:engram.forward=method
                for engram,rows,hashes in saved_engram:
                    engram._rows=None;engram._hash_ids=hashes
                if decode:
                    rebuilt=runner.model_state.prepare_attn(batch,CUDAGraphMode.NONE,
                        self._p21_raw_tables,self._p21_raw_slots,runner.attn_groups,runner.kv_cache_config,
                        for_capture=False)
                    original_metadata.clear();original_metadata.update(rebuilt)
                    ctx.attn_metadata=original_metadata
        # Suffix resumes at a folded HC state, so no second hc_post at layer21.
        residual=post=res=None;aux=[]
        suffixctx=copy.copy(ctx);suffixctx.additional_kwargs=dict(ctx.additional_kwargs)
        suffixctx.all_moe_layers=None;suffixctx.moe_layer_index=0
        if hasattr(suffixctx,'_ilu_gathered_router_padding'):del suffixctx._ilu_gathered_router_padding
        with override_forward_context(suffixctx):
            for index in range(21,40):
                h,residual,post,res,pre=model.layers[index](h,positions,input_ids,pre,post,res,residual)
                if index+1 in model.aux_hidden_state_layers:
                    aux.append(mhc.hc_post(h,residual,post,res).mean(dim=1))
        h=mhc.hc_post(h,residual,post,res)
        h=model.norm(mhc.hc_pre(h,pre))
        self._p21_counters['GO']+=1
        self._p21_eager_warmed |= any(int(v[1]) for v in votes)
        replay_rows=[state['N']-packet['metadata']['S'] for _,state,packet,_ in tails]
        self._p21_counters['skipped_tail_rows']+=sum(replay_rows)
        for rows in replay_rows:self._p21_counters[f'tail{rows}_requests']+=1
        save(self._p21_result/f'go_{self._p21_rank}.json',{'rank':self._p21_rank,
             'request_ids':[state['id'] for _,state,_,_ in tails],
             'request_replay_rows':replay_rows,
             'decode_rows':decode,'global_votes':[v.tolist() for v in votes],
             'prefix_tokens':prefix_tokens,'max_num_seqs':self._p21_max_reqs,
             'counters':self._p21_counters,'GPU_packet_bytes':0,'new_graphs':0,
             'max_model_len':self.vllm_config.model_config.max_model_len})
        return (h,aux) if aux else h

"""CPU evidence: successful original receive + replay-ready fullN + current pages."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .source_state import local_paths
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value)+'\n');tmp.replace(path)
def selected(group):
    return any('.layers.' in name and int(name.split('.layers.')[1].split('.')[0]) in (2,8,14,20)
               and (name.endswith('.attn') or name.endswith('.attn.indexer.k_cache')) for name in group.layer_names)
def install():
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.opt_nixl.scheduler import NixlConnectorScheduler
    if getattr(Scheduler,'_p21_full_ready_installed',False):return
    Scheduler._p21_full_ready_installed=True
    old_update=Scheduler._update_waiting_for_remote_kv;old_mark=Scheduler._mark_prefix_replay
    old_init=NixlConnectorScheduler.__init__;old_alloc=NixlConnectorScheduler.update_state_after_alloc
    old_build=NixlConnectorScheduler.build_connector_meta
    def initialized(self,*args,**kwargs):
        value=old_init(self,*args,**kwargs)
        self._p21_wire, self._p21_nonce = local_paths(self.engine_id)
        self._p21_remote_endpoints={}
        self._p21_requests={};self._p21_selected=[g for g,group in enumerate(self.kv_cache_config.kv_cache_groups) if selected(group)]
        if self.is_consumer:
            save(self._p21_wire/f'engine_installed_{self.vllm_config.parallel_config.data_parallel_index}.json',
                 dict(pid=os.getpid(),run_uuid=self._p21_nonce,engine_id=self.engine_id,groups=self._p21_selected,
                      successful_receive_scope=True,original_cache_publication_unchanged=True))
        return value
    def update(self,request):
        scope=(request.request_id in self.finished_recving_kv_req_ids and request.request_id not in self.failed_recving_kv_req_ids)
        previous=getattr(self,'_p21_success_scope',None)
        self._p21_success_scope=(request,scope)
        try:return old_update(self,request)
        finally:self._p21_success_scope=previous
    def mark(self,request,num_hit_tokens):
        value=old_mark(self,request,num_hit_tokens)
        scope=getattr(self,'_p21_success_scope',None)
        N=request.num_prompt_tokens
        if scope is not None and scope[0] is request and scope[1] and N>=128 and N%2==0 and num_hit_tokens==N and value==128:
            record=getattr(request,'_p21_alloc_receipt',None)
            if record and record['external']>0:
                request._p21_ready_receipt=dict(N=N,fullkv_valid_before_replay=N,original_receive_success=True,
                    alloc_serial=record['serial'],token_sha256=hashlib.sha256(np.asarray(request.prompt_token_ids,dtype=np.int32).tobytes()).hexdigest(),
                    mark_mono_ns=time.monotonic_ns())
                request._p21_scheduler=self
        return value
    def allocated(self,request,blocks,num_external_tokens):
        if self.is_consumer and num_external_tokens>0:
            serial=getattr(request,'_p21_alloc_serial',0)+1;request._p21_alloc_serial=serial
            positions={g:[dict(logical=j,block_id=block.block_id) for j,block in enumerate(blocks.blocks[g])
                          if block.block_hash is None and not block.is_null] for g in self._p21_selected}
            request._p21_alloc_receipt=dict(serial=serial,external=num_external_tokens,
                local=request.num_computed_tokens,received_positions=positions,
                remote_engine_id=request.kv_transfer_params.get('remote_engine_id'))
            request._p21_ready_receipt=None
            self._p21_requests[request.request_id]=request
        value=old_alloc(self,request,blocks,num_external_tokens)
        if self.is_consumer and num_external_tokens>0:
            remote_id=request.kv_transfer_params.get('remote_engine_id')
            peer=self.peer_node_metas.get(remote_id)
            if peer is not None:
                self._p21_remote_endpoints[request.request_id]=dict(engine_id=remote_id,addr=peer.addr)
        return value
    def build(self,output):
        metadata=old_build(self,output);receipts={}
        for rid in tuple(self._p21_requests):
            request=self._p21_requests[rid]
            if rid in output.finished_req_ids or rid in output.preempted_req_ids:
                request._p21_ready_receipt=None;self._p21_requests.pop(rid,None)
                self._p21_remote_endpoints.pop(rid,None);continue
            ready=getattr(request,'_p21_ready_receipt',None)
            if ready is None or rid not in output.num_scheduled_tokens:continue
            if request.num_computed_tokens!=ready['N']-128:continue
            scheduler=request._p21_scheduler
            current=scheduler.kv_cache_manager.get_blocks(rid).get_block_ids(group_ids=self._p21_selected)
            record=request._p21_alloc_receipt
            receipts[rid]=dict(request_id=rid,run_uuid=self._p21_nonce,engine_id=self.engine_id,**ready,
                external=record['external'],local=record['local'],groups={g:list(ids) for g,ids in zip(self._p21_selected,current)},
                received_positions=record['received_positions'],remote_engine_id=record['remote_engine_id'],
                group_specs={g:dict(block_size=self.kv_cache_config.kv_cache_groups[g].kv_cache_spec.block_size,
                                   layer_names=list(self.kv_cache_config.kv_cache_groups[g].layer_names),
                                   dtype=str(self.kv_cache_config.kv_cache_groups[g].kv_cache_spec.dtype)) for g in self._p21_selected})
        metadata.source_full_ready_receipts=receipts
        metadata.source_state_endpoints={rid:self._p21_remote_endpoints[rid]
            for rid in metadata.reqs_to_recv if rid in self._p21_remote_endpoints}
        return metadata
    NixlConnectorScheduler.__init__=initialized
    NixlConnectorScheduler.update_state_after_alloc=allocated
    NixlConnectorScheduler.build_connector_meta=build
    Scheduler._update_waiting_for_remote_kv=update;Scheduler._mark_prefix_replay=mark

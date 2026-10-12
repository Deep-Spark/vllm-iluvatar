"""Original DP proxy plus bounded current-request complete CPU source join."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import types
ROOT=Path(os.environ.get('DSV41_WORKSPACE',str(Path(__file__).resolve().parents[5])))
WIRE=Path(os.environ['P21_WIRE']);RESULT=Path(os.environ['P21_SHADOW_DIR']);RUN_UUID=os.environ['P21_RUN_UUID']
def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value)+'\n');tmp.replace(path)
module=types.ModuleType('source_original_proxy');sys.modules[module.__name__]=module
namespace=module.__dict__;namespace['__file__']=str(ROOT/'run_script/proxy.py')
exec(compile((ROOT/'run_script/proxy.py').read_text(),namespace['__file__'],'exec'),namespace)
original_send=namespace['send_prefill_request_to_service']
lock=threading.Lock();pending={};retired={};epoch=0
counts=dict(requests=0,lookups=0,hit=0,fresh=0,reused=0,unavailable=0,evictions=0)
class OwnedRequests(set):
    def discard(self,rid):
        super().discard(rid)
        with lock:retired['chatcmpl-'+rid]=time.monotonic_ns()
namespace['running_request_ids']=OwnedRequests(namespace['running_request_ids'])
async def send(*args,**kwargs):
    global epoch
    rid='chatcmpl-'+(args[3] if len(args)>3 else kwargs['request_id'])
    with lock:
        epoch+=1
        pending[rid]=dict(epoch=epoch,request_id=rid,run_uuid=RUN_UUID,created_mono_ns=time.monotonic_ns(),done_mono_ns=None)
        counts['requests']+=1
    try:return await original_send(*args,**kwargs)
    finally:
        with lock:
            if rid in pending:pending[rid]['done_mono_ns']=time.monotonic_ns()
namespace['send_prefill_request_to_service']=send

def join(current,rows):
    rid=current['request_id'];N=rows[0]['N'];key=rows[0]['token_sha256']
    assert all(r['run_uuid']==RUN_UUID and r['id']==rid and r['N']==N and r['token_sha256']==key for r in rows)
    target=WIRE/rid;target.mkdir(exist_ok=True)
    for tp in range(2):
        stages=sorted((r for r in rows if r['tp']==tp),key=lambda r:r['pp']);assert len(stages)==4
        fields=[];parts=[];offset=0
        for row in stages:
            raw=(RESULT/rid/f"rank{row['rank']}.bin").read_bytes();assert hashlib.sha256(raw).hexdigest()==row['sha256']
            parts.append(raw);fields.extend({**rec,'offset_bytes':offset+rec['offset_bytes']} for rec in row['records']);offset+=len(raw)
        raw=b''.join(parts);assert len(raw)==9504768
        (target/f'packet_tp{tp}.bin').write_bytes(raw)
        save(target/f'packet_tp{tp}.json',dict(request_id=rid,run_uuid=RUN_UUID,epoch=current['epoch'],tp=tp,N=N,H=N,S=N-128,
            token_sha256=key,byte_count=len(raw),sha256=hashlib.sha256(raw).hexdigest(),fields=fields,source_complete=True,
            lens_valid=False,source_generation=stages[-1]['generation'],published_mono_ns=time.monotonic_ns()))
    save(target/'current_source.json',dict(request_id=rid,run_uuid=RUN_UUID,epoch=current['epoch'],N=N,token_sha256=key,
        fresh=True,computed=min(s['computed'] for r in rows for s in r['spans']),spans=rows[0]['spans'],ready_mono_ns=time.monotonic_ns()))
    save(WIRE/f"published_{current['epoch']}.json",dict(epoch=current['epoch'],request_id=rid,N=N,fresh=True))

def publish():
    report_time=0
    while True:
        with lock:requests=[dict(r) for r in pending.values()]
        for current in requests:
            rid=current['request_id'];rows=[]
            for rank in range(8):
                try:rows.append(json.loads((RESULT/rid/f'auto_{rank}.json').read_text()))
                except (FileNotFoundError,json.JSONDecodeError):pass
            if len(rows)==8:
                if all(r['fresh'] and not r['abort'] for r in rows):join(current,rows);counts['fresh']+=1
                else:
                    counts['unavailable']+=1
                    save(WIRE/f"unavailable_{current['epoch']}.json",dict(request_id=rid,reason='currentP_incomplete128',spans=[r['spans'] for r in rows]))
                with lock:pending.pop(rid,None)
                shutil.rmtree(RESULT/rid,ignore_errors=True)
            elif current['done_mono_ns'] and time.monotonic_ns()-current['done_mono_ns']>500_000_000:
                counts['unavailable']+=1
                save(WIRE/f"unavailable_{current['epoch']}.json",dict(request_id=rid,ready_ranks=len(rows)))
                with lock:pending.pop(rid,None)
        if time.monotonic()-report_time>=1:
            save(WIRE/'source_counters.json',{**counts,'cache_keys':0,'cache_owned_bytes':0,'pending':len(pending),'run_uuid':RUN_UUID,'current_source_only':True})
            report_time=time.monotonic()
        with lock:
            cleanup=[rid for rid,when in retired.items() if time.monotonic_ns()-when>2_000_000_000]
            for rid in cleanup:retired.pop(rid,None)
        for rid in cleanup:
            for path in (WIRE/rid).glob('packet_tp*.bin'):path.unlink(missing_ok=True)
        time.sleep(.002)
def main():
    threading.Thread(target=publish,daemon=True,name='P21-current-publisher').start()
    parser=argparse.ArgumentParser();namespace['register_command'](parser.add_subparsers(dest='subcommand'))
    args=parser.parse_args(['proxy_nixl_push',*sys.argv[1:]]);args.func(args)

if __name__=='__main__':main()

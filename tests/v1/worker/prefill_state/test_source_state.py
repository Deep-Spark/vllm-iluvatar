"""State transport, current-request identity, publication and workspace contracts."""

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest

import msgspec
import torch
import zmq

from vllm_iluvatar.v1.worker.prefill_state import source_state as pub
from vllm_iluvatar.v1.worker.prefill_state.decode_prefix import PrefixBuffers
from vllm_iluvatar.v1.worker.prefill_state.cache_rows import (
    copy_indexer_row_to_host, restore_indexer_row,
)


class PacketTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.wire=Path(self.temp.name)/'producer'
        self.source=self.wire/'source'
        self.source.mkdir(parents=True)
        self.engine='producer-engine'
        self.nonce='producer-service-nonce'
        self.digest=hashlib.sha256(b'exact-prompt').hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def shards(self, rid='chatcmpl-request', n=262144, indexer_bytes=132):
        directory=pub.request_dir(self.source,rid);directory.mkdir()
        hit,start,q=pub.replay_layout(n)
        rows=[]
        for rank in range(8):
            pp,tp=divmod(rank,2)
            records=[]
            for layer in range(pp*5,pp*5+(5 if pp<3 else 6)):
                records.append(dict(name=f'layers.{layer}.swa',kind='SWA',layer=layer,
                                    shape=[q,512],dtype='torch.bfloat16'))
            if pp<3:
                records.append(dict(name=f'layers.{(2,8,14)[pp]}.ring',kind='ring',
                                    layer=(2,8,14)[pp],shape=[16,1024],dtype='torch.float32'))
            else:
                for name,shape,dtype in (
                    ('HC',[q,4,5120],'torch.bfloat16'),('pre_mix',[q,4],'torch.float32'),
                    ('topk20',[q,512],'torch.int32'),('candidates20',[q,2048],'torch.int32')):
                    records.append(dict(name=name,kind='boundary21',shape=shape,dtype=dtype))
                if q==129:
                    records.extend([
                        dict(name='main20',kind='main_tail',layer=20,shape=[1,512],dtype='torch.bfloat16'),
                        dict(name='indexer20',kind='indexer_tail',layer=20,shape=[indexer_bytes],dtype='torch.uint8'),
                    ])
            offset=0
            for record in records:
                size=pub.DTYPES[record['dtype']][1]
                for dim in record['shape']:size*=dim
                record.update(offset_bytes=offset,byte_count=size);offset+=size
            raw=bytes([rank+1])*offset
            (directory/f'rank{rank}.bin').write_bytes(raw)
            row=dict(id=rid,engine_id=self.engine,run_uuid=self.nonce,N=n,H=hit,S=start,
                     token_sha256=self.digest,rank=rank,pp=pp,tp=tp,fresh=True,abort=False,
                     generation=rank+1,byte_count=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                     records=records,spans=[dict(computed=start,query=q)])
            pub.save(directory/f'auto_{rank}.json',row);rows.append(row)
        return rows

    def publisher(self,**kw):
        return pub.StatePublisher(self.wire,self.engine,self.nonce,**kw)

    def request(self,rid='chatcmpl-request',tp=0,n=262144):
        return (pub.GET_STATE,self.engine,rid,tp,n,self.digest)

    def test_odd_prompt_carries129_rows_and_uncached_cr1_boundary(self):
        for n in (32769,76393,131071):
            for width in (132,256):
                with self.subTest(n=n,indexer_bytes=width):
                    rid=f'odd-{n}-{width}'
                    self.shards(rid,n,indexer_bytes=width);self.publisher().poll(10)
                    reply,raw=pub.serve_state(self.wire,self.engine,self.request(rid,n=n))
                    self.assertEqual(reply['status'],'READY')
                    self.assertEqual(len(raw),pub.packet_bytes(n,width))
                    meta=reply['metadata'];self.assertEqual((meta['H'],meta['S']),(n-1,n-129))
                    pub.validate_fields(meta['fields'],len(raw),n)
                    self.assertTrue(any(f['kind']=='main_tail' for f in meta['fields']))
                    for kind in ('main_tail','indexer_tail'):
                        fields=[f for f in meta['fields'] if f['kind']!=kind]
                        with self.assertRaises(ValueError):pub.validate_fields(fields,len(raw),n)

    def test_replay_geometry_rejects_stale_odd_prefix_boundary(self):
        rows=self.shards('bad-odd-H',76393)
        rows[0]['H']=76393
        with self.assertRaisesRegex(ValueError,'capture mismatch'):
            pub.join_packets(self.wire,self.source,self.engine,self.nonce,'bad-odd-H',1,rows)
        for n in (0,127,True,128.0):
            with self.assertRaises(ValueError):pub.replay_layout(n)

    def test_atomic_two_tp_completion_and_exact_wire_fields(self):
        rows=self.shards();rid=rows[0]['id'];directory=pub.request_dir(self.source,rid)
        missing=directory/'auto_7.json';missing.unlink()
        publisher=self.publisher();publisher.poll(10)
        self.assertEqual(pub.serve_state(self.wire,self.engine,self.request())[0]['status'],'PENDING')
        pub.save(missing,rows[7]);publisher.poll(11)
        self.assertEqual(publisher.counts['fresh'],1)
        self.assertFalse(directory.exists())
        for tp in range(2):
            reply,raw=pub.serve_state(self.wire,self.engine,self.request(tp=tp))
            self.assertEqual(reply['status'],'READY');self.assertEqual(len(raw),pub.PACKET_BYTES)
            self.assertEqual(raw[:1],bytes([tp+1]));self.assertEqual(raw[-1:],bytes([tp+7]))
            pub.validate_fields(reply['metadata']['fields'],len(raw))

    def test_bad_shard_does_not_publish_or_poison_another_request(self):
        bad=self.shards('bad');good=self.shards('good')
        pub.save(pub.request_dir(self.source,'bad')/'auto_7.json',{**bad[7],'sha256':'bad'})
        publisher=self.publisher();publisher.poll(10)
        self.assertEqual(publisher.counts['fresh'],1);self.assertEqual(publisher.counts['errors'],1)
        self.assertFalse((pub.request_dir(self.wire,'bad')/'current_source.json').exists())
        self.assertTrue((pub.request_dir(self.wire,good[0]['id'])/'current_source.json').exists())

    def test_engine_nonce_request_hash_and_rank_mismatch_reject(self):
        for name in ('engine_id','run_uuid','id','token_sha256','rank','generation','N'):
            with self.subTest(name=name):
                rows=self.shards('bad-'+name)
                rows[7][name]=0 if name in ('rank','generation','N') else 'wrong'
                with self.assertRaises(ValueError):
                    pub.join_packets(self.wire,self.source,self.engine,self.nonce,rows[0]['id'],1,rows)
                self.assertFalse((pub.request_dir(self.wire,rows[0]['id'])/'current_source.json').exists())

    def test_partial_and_timeout_fail_closed_with_bounded_cleanup(self):
        rows=self.shards('partial')
        pub.save(pub.request_dir(self.source,'partial')/'auto_3.json',{**rows[3],'fresh':False})
        rows=self.shards('timeout');(pub.request_dir(self.source,'timeout')/'auto_7.json').unlink()
        publisher=self.publisher(shard_timeout_s=2,packet_ttl_s=3)
        publisher.poll(10);self.assertEqual(publisher.counts['unavailable'],1)
        publisher.poll(13);self.assertEqual(publisher.counts['unavailable'],2)
        publisher.poll(17)
        self.assertEqual(len(publisher.pending),0);self.assertEqual(len(publisher.retired),0)
        self.assertFalse(pub.request_dir(self.source,'timeout').exists())

    def test_field_shape_dtype_overlap_and_missing_records_reject(self):
        rows=self.shards();self.publisher().poll(10)
        reply,_=pub.serve_state(self.wire,self.engine,self.request())
        for change in ('shape','dtype','offset_bytes','duplicate','missing','duplicate_name',
                       'float_shape','float_bytes','bool_offset','bool_layer'):
            fields=json.loads(json.dumps(reply['metadata']['fields']))
            if change=='duplicate':fields.append(fields[0])
            elif change=='missing':fields.pop()
            elif change=='duplicate_name':fields[1]['name']=fields[0]['name']
            elif change=='shape':fields[0]['shape']=[128,511]
            elif change=='float_shape':fields[0]['shape']=[128.0,512]
            elif change=='float_bytes':fields[0]['byte_count']=float(fields[0]['byte_count'])
            elif change=='bool_offset':fields[0]['offset_bytes']=False
            elif change=='bool_layer':fields[0]['layer']=False
            elif change=='dtype':fields[0]['dtype']='I32'
            else:fields[0]['offset_bytes']=1
            with self.subTest(change=change),self.assertRaises(ValueError):
                pub.validate_fields(fields,pub.PACKET_BYTES)

    def test_request_id_cannot_escape_local_directory(self):
        rid='../some/../../request'
        self.shards(rid);self.publisher().poll(10)
        self.assertEqual(pub.request_dir(self.wire,rid).parent,self.wire)
        self.assertEqual(pub.serve_state(self.wire,self.engine,self.request(rid=rid))[0]['status'],'READY')

    def test_transport_without_shared_consumer_filesystem(self):
        self.shards();self.shards('odd',76393);self.publisher().poll(10)
        # Real TCP frames with a separate empty D root; only P opens packet files.
        consumer_root=Path(self.temp.name)/'consumer';consumer_root.mkdir()
        context=zmq.Context();server=context.socket(zmq.ROUTER);server.linger=0
        port=server.bind_to_random_port('tcp://127.0.0.1');server.rcvtimeo=3000
        errors=[]
        def serve():
            try:
                for _ in range(6):
                    identity,empty,data=server.recv_multipart();self.assertEqual(empty,b'')
                    reply,raw=pub.serve_state(self.wire,self.engine,msgspec.msgpack.decode(data))
                    server.send_multipart((identity,b'',msgspec.msgpack.encode(reply),raw))
            except Exception as error:errors.append(error)
        thread=threading.Thread(target=serve);thread.start()
        client=context.socket(zmq.REQ);client.linger=0;client.rcvtimeo=3000
        client.connect(f'tcp://127.0.0.1:{port}')
        try:
            for tp in (0,1):
                reply,raw=pub.fetch_state(client,self.engine,'chatcmpl-request',tp,262144,self.digest)
                self.assertEqual(reply['status'],'READY');self.assertEqual(len(raw),pub.PACKET_BYTES)
                reply,raw=pub.fetch_state(client,self.engine,'odd',tp,76393,self.digest)
                self.assertEqual(reply['status'],'READY')
                self.assertEqual((reply['metadata']['H'],reply['metadata']['S']),(76392,76264))
                self.assertEqual(len(raw),pub.packet_bytes(76393))
            with self.assertRaisesRegex(ValueError,'endpoint identity'):
                pub.fetch_state(client,'wrong-engine','chatcmpl-request',0,262144,self.digest)
            reply,_=pub.fetch_state(client,self.engine,'chatcmpl-request',0,262144,'0'*64)
            self.assertEqual(reply['status'],'IDENTITY_MISMATCH')
            self.assertEqual(list(consumer_root.iterdir()),[])
        finally:
            thread.join(4);client.close();server.close();context.term()
        self.assertFalse(thread.is_alive());self.assertEqual(errors,[])

    def test_packet_ttl_is_measured_from_publication(self):
        rows=self.shards();missing=pub.request_dir(self.source,rows[0]['id'])/'auto_7.json';missing.unlink()
        publisher=self.publisher(shard_timeout_s=100,packet_ttl_s=5)
        publisher.poll(10);pub.save(missing,rows[7]);publisher.poll(50);publisher.poll(54)
        self.assertTrue((pub.request_dir(self.wire,rows[0]['id'])/'current_source.json').exists())
        publisher.poll(56);self.assertFalse(pub.request_dir(self.wire,rows[0]['id']).exists())

    def test_reused_request_id_cannot_read_the_previous_admission(self):
        self.shards();publisher=self.publisher();publisher.poll(10)
        old,_=pub.serve_state(self.wire,self.engine,self.request())
        rows=self.shards()
        for row in rows:
            row['generation']+=1
            pub.save(pub.request_dir(self.source,row['id'])/f"auto_{row['rank']}.json",row)
        # Admission blocks the old packet before the publisher's next poll.
        self.assertEqual(pub.serve_state(self.wire,self.engine,self.request())[0]['status'],'PENDING')
        publisher.poll(11)
        new,_=pub.serve_state(self.wire,self.engine,self.request())
        self.assertEqual(new['status'],'READY')
        self.assertGreater(new['metadata']['epoch'],old['metadata']['epoch'])
        self.assertGreater(new['metadata']['source_generation'],old['metadata']['source_generation'])


class BoundaryRowTest(unittest.TestCase):
    def test_quantized_page_key_and_scale_are_separate_with_page_padding(self):
        rows=64
        # Same132-byte nominal view and256-byte allocated page pitch as INT8.
        source=torch.full((3,rows*256),0xA7,dtype=torch.uint8)
        destination=torch.full_like(source,0x5D)
        src=source.as_strided((3,rows,132),(rows*256,132,1))
        dst=destination.as_strided((3,rows,132),(rows*256,132,1))
        for row in (0,1,rows-1):
            with self.subTest(row=row):
                source[1,row*128:(row+1)*128]=torch.arange(128,dtype=torch.uint8)
                source[1,rows*128+row*4:rows*128+(row+1)*4]=torch.tensor([1,2,3,4],dtype=torch.uint8)
                before=destination.clone()
                packet=torch.empty(132,dtype=torch.uint8)
                copy_indexer_row_to_host(packet,src,1,row)
                self.assertEqual(packet[:128].tolist(),list(range(128)))
                self.assertEqual(packet[128:].tolist(),[1,2,3,4])
                restore_indexer_row(dst,1,row,packet)
                expected=before.clone()
                expected[1,row*128:(row+1)*128]=packet[:128]
                expected[1,rows*128+row*4:rows*128+(row+1)*4]=packet[128:]
                self.assertTrue(torch.equal(destination,expected))

    def test_bf16_boundary_row_at_new_page_preserves_all_other_rows(self):
        source=torch.arange(3*64*128,dtype=torch.float32).to(torch.bfloat16).reshape(3,64,128)
        destination=torch.zeros_like(source)
        packet=torch.empty(256,dtype=torch.uint8)
        copy_indexer_row_to_host(packet,source,2,0)
        restore_indexer_row(destination,2,0,packet)
        self.assertTrue(torch.equal(destination[2,0],source[2,0]))
        self.assertEqual(torch.count_nonzero(destination[:2]).item(),0)
        self.assertEqual(torch.count_nonzero(destination[2,1:]).item(),0)
        with self.assertRaises(ValueError):restore_indexer_row(destination,2,0,packet[:132])


class WorkspaceTest(unittest.TestCase):
    def test_configured_size_alignment_disjoint_views_and_initialized_padding(self):
        for seqs in (1,2,3,4,8,16):
            with self.subTest(seqs=seqs):
                buffers=PrefixBuffers(seqs*6,5120,4,torch.device('cpu'))
                self.assertEqual(buffers.owner.numel(),seqs*6*40989)
                fields=(buffers.hidden,buffers.pre_mix,buffers.positions,buffers.input_ids,buffers.is_padding)
                regions=[]
                for field in fields:
                    self.assertEqual(field.untyped_storage().data_ptr(),buffers.owner.data_ptr())
                    self.assertEqual(field.data_ptr()%field.element_size(),0)
                    regions.append((field.data_ptr(),field.data_ptr()+field.numel()*field.element_size()))
                self.assertTrue(all(a[1]<=b[0] for a,b in zip(regions,regions[1:])))
                view=buffers.for_tokens(6)
                view.hidden.zero_();view.pre_mix.zero_();view.positions.zero_();view.input_ids.zero_()
                view.is_padding.fill_(True);view.is_padding[:3].fill_(False)
                self.assertEqual(view.is_padding.tolist(),[False]*3+[True]*3)
                self.assertEqual(view.hidden.data_ptr(),buffers.hidden.data_ptr())
                with self.assertRaises(ValueError):buffers.for_tokens(seqs*6+1)


if __name__=='__main__':unittest.main()

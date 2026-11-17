"""Opt-in DSV4.1 TP2/DP4 decode dispatch with one packed FP8 AllGather."""
import os
import torch
import triton
import triton.language as tl
import ixformer.distributed as ixfd
import ixformer.inference.functions as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger

logger = init_logger(__name__)
_IDENTITIES = {}

@triton.jit
def _pack(Q,S,I,P,O,H:tl.constexpr,STRIDE:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    q=tl.load(Q+row*H+col,col<H,other=0)
    s=tl.load(S+row*4+col-H,(col>=H)&(col<H+4),other=0)
    i=tl.load(I+row*24+col-H-4,(col>=H+4)&(col<H+28),other=0)
    p=tl.load(P+row*24+col-H-28,(col>=H+28)&(col<STRIDE),other=0)
    tl.store(O+row*STRIDE+col,tl.where(col<H,q,tl.where(col<H+4,s,tl.where(col<H+28,i,p))),col<STRIDE)

@triton.jit
def _unpack(A,S,I,P,H:tl.constexpr,STRIDE:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,64)
    value=tl.load(A+row*STRIDE+H+col,col<52,other=0)
    tl.store(S+row*4+col,value,col<4)
    tl.store(I+row*24+col-4,value,(col>=4)&(col<28))
    tl.store(P+row*24+col-28,value,(col>=28)&(col<52))

@triton.jit
def _expand(Q,S,MAP,O,OS,H:tl.constexpr,K:tl.constexpr,B:tl.constexpr,QS:tl.constexpr):
    row=tl.program_id(0);col=tl.program_id(1)*B+tl.arange(0,B)
    route=tl.load(MAP+row);token=tl.maximum(route,0)//K
    value=tl.load(Q+token*QS+col,(route>=0)&(col<H),other=0)
    tl.store(O+row*H+col,value,col<H)
    if tl.program_id(1)==0:
        tl.store(OS+row,tl.load(S+token,route>=0,other=1.))

@triton.jit
def quant_pack_routes(X,I,P,O,H:tl.constexpr,STRIDE:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,B)
    x=tl.load(X+row*H+col,col<H,other=0).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(x),0)*(1.0/448.0),1.0e-12)
    q=tl.minimum(tl.maximum(tl.div_rn(x,scale),-448.),448.).to(tl.float8e4nv)
    tl.store((O+row*STRIDE).to(tl.pointer_type(tl.float8e4nv))+col,q,col<H)
    tl.store((O+row*STRIDE+H).to(tl.pointer_type(tl.float32)),scale)
    j=tl.arange(0,8)
    ids=tl.load(I+row*6+j,j<6,other=0)
    probs=tl.load(P+row*6+j,j<6,other=0)
    tl.store((O+row*STRIDE+H+4).to(tl.pointer_type(tl.int32))+j,ids,j<6)
    tl.store((O+row*STRIDE+H+28).to(tl.pointer_type(tl.float32))+j,probs,j<6)

def expand_prequantized_input(q,scales,mapping,topk):
    rows=mapping.numel();hidden=q.shape[1]
    output=torch.empty((rows,hidden),device=q.device,dtype=torch.float8_e4m3fn)
    output_scales=torch.empty(rows,device=q.device,dtype=torch.float32)
    _expand[(rows,triton.cdiv(hidden,1024))](q.view(torch.uint8),scales,mapping,
        output.view(torch.uint8),output_scales,hidden,topk,1024,q.stride(0))
    return output,output_scales

def maybe_packed_experts(owner,x,input_ids,ctx,group,sizes):
    runner=owner.experts;layer=runner.routed_experts
    communicator=group.device_communicator
    # Every collective decision uses replicated configuration/global DP state.
    if not (
        type(owner).__module__ in ('vllm_iluvatar.models.deepseek_v4.model',
                                   'vllm_iluvatar.models.deepseek_v41.model')
        and type(runner._quant_method).__name__=='IluMxfp4MoEMethod'
        and ctx.dp_metadata is not None and ctx.dp_metadata.all_ranks_decode_only
        and not runner.moe_config.is_sequence_parallel
        and get_tensor_model_parallel_world_size()==2 and len(sizes)==4
        and all(n==sizes[0] for n in sizes) and 24<=sizes[0]<=96
        and x.dtype==torch.bfloat16 and x.shape[1]==5120
        and getattr(runner.moe_config,'experts_per_token',0)==6
        and layer.w13_weight.dtype==torch.uint8
        and getattr(layer,'w13_bias',None) is None and getattr(layer,'w2_bias',None) is None
        and communicator is not None and communicator.ixfd_group is not None
        and not getattr(ctx,'_ilu_is_padding_gathered',False)
    ):return None
    # A rank-local stride must not choose a different collective from peers.
    # Normalize locally instead of taking a rank-local BF16 fallback.
    x=x.contiguous()
    from .ilu_mxfp4_fused_moe import _ilu_mxfp4_fused_experts_ixformer
    local_n=x.shape[0];total=sum(sizes)
    gate=runner.gate if runner.gate is not None else owner.gate
    logits,_=gate(x)
    # Keep the local padding mask and local hash/vision token IDs here.
    local_probs,local_ids=runner.router.select_experts(hidden_states=x,router_logits=logits,
        topk_indices_dtype=torch.int32,input_ids=owner._router_input_ids(runner,input_ids))
    local_ids=local_ids.to(torch.int32).contiguous();local_probs=local_probs.float().contiguous()
    packet=torch.empty((local_n,5172),device=x.device,dtype=torch.uint8)
    gathered=torch.empty((total,5172),device=x.device,dtype=torch.uint8)
    if os.environ.get('VLLM_ILU_V41_PACKED_FP8_FUSED_QUANT','0')=='1':
        quant_pack_routes[(local_n,)](x,local_ids,local_probs,packet,5120,5172,8192,num_warps=8)
    else:
        key=(x.device,local_n)
        if key not in _IDENTITIES:_IDENTITIES[key]=torch.arange(local_n,device=x.device,dtype=torch.int32)
        q,scales=ops.moe_expand_input_dynamic_scaled_fp8(x,_IDENTITIES[key],1)
        _pack[(local_n,6)](q.view(torch.uint8),scales.view(torch.uint8),local_ids.view(torch.uint8),local_probs.view(torch.uint8),packet,5120,5172,1024)
    ixfd.all_gather_into_tensor(gathered,packet,group=communicator.ixfd_group,async_op=True)
    full_scales=torch.empty(total,device=x.device,dtype=torch.float32)
    ids=torch.empty((total,6),device=x.device,dtype=torch.int32)
    probs=torch.empty((total,6),device=x.device,dtype=torch.float32)
    _unpack[(total,)](gathered,full_scales.view(torch.uint8),ids.view(torch.uint8),probs.view(torch.uint8),5120,5172)
    logger.info_once('V4.1 packed FP8 dispatch active')
    # Only shape/device/dtype are consumed when prequantized input is provided.
    shape_view=x[:1].expand(total,-1)
    mode=1 if os.environ.get('VLLM_ILU_V41_DECODE_MOE_MODE1','0')=='1' and total<=768 else None
    return _ilu_mxfp4_fused_experts_ixformer(shape_view,layer.w13_weight,layer.w13_weight_scale,
        layer.w2_weight,layer.w2_weight_scale,probs,ids,swiglu_limit=layer.swiglu_limit,
        expert_map=layer.expert_map,output_scale=getattr(layer,'_ilu_output_scale',1.),extra_residual=None,
        w13_bias=None,w2_bias=None,gemm_backend='ixmm',gemm_mode=mode,swiglu_variant='dsv4_clamp',
        quantized_input=(gathered[:,:5120].view(torch.float8_e4m3fn),full_scales))

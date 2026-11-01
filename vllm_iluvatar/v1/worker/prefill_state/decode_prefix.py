"""Configured E25 prefix workspace, independent of target and draft graphs."""

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class PrefixViews:
    hidden: torch.Tensor
    pre_mix: torch.Tensor
    input_ids: torch.Tensor
    positions: torch.Tensor
    is_padding: torch.Tensor


class PrefixBuffers:
    """One small owner sized from the configured decode budget, not batch four.

    Allocate before memory profiling so KV planning includes this workspace.
    Every use initializes its active rows. Metadata/table/slot views continue to
    refer to the runner's existing buffers; no second graph pool is captured.
    """

    def __init__(self, max_tokens: int, hidden_size: int, hc_mult: int,
                 device: torch.device):
        if max_tokens<=0 or hidden_size<=0 or hc_mult<=0:
            raise ValueError('Prefix dimensions must be positive')
        self.max_tokens=max_tokens
        shapes=(('hidden', (max_tokens,hc_mult,hidden_size), torch.bfloat16),
                ('pre_mix', (max_tokens,hc_mult), torch.float32),
                ('positions', (max_tokens,), torch.int64),
                ('input_ids', (max_tokens,), torch.int32),
                ('is_padding', (max_tokens,), torch.bool))
        offsets={};offset=0
        for name,shape,dtype in shapes:
            element_size=torch.empty((),dtype=dtype).element_size()
            offset=(offset+element_size-1)//element_size*element_size
            size=math.prod(shape)*element_size
            offsets[name]=(offset,size)
            offset+=size
        self.owner=torch.empty(offset,dtype=torch.uint8,device=device)
        for name,shape,dtype in shapes:
            start,size=offsets[name]
            setattr(self,name,self.owner[start:start+size].view(dtype).view(shape))

    def for_tokens(self, tokens: int) -> PrefixViews:
        if not 0<tokens<=self.max_tokens:
            raise ValueError('Prefix exceeds its configured workspace')
        return PrefixViews(self.hidden[:tokens],self.pre_mix[:tokens],
                           self.input_ids[:tokens],self.positions[:tokens],self.is_padding[:tokens])

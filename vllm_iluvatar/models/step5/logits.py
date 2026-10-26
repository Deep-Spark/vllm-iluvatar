"""Step5 TP logits: gather contiguous rank blocks, then reorder natively."""

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor


class Step5LogitsProcessor(LogitsProcessor):
    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if (
            not self.use_all_gather
            or not logits.is_cuda
            or logits.ndim != 2
            or (logits.dtype not in (torch.bfloat16, torch.float16, torch.float32))
        ):
            return super()._gather_logits(logits)
        from ixformer.inference.functions.models.step5.logits.float.logits_reorder import (
            logits_reorder,
        )

        ranks = get_tensor_model_parallel_world_size()
        gathered = tensor_model_parallel_all_gather(logits, dim=0)
        if ranks == 1 or logits.shape[0] == 1:
            return gathered.view(logits.shape[0], ranks * logits.shape[1])[
                :, : self.org_vocab_size
            ]
        return logits_reorder(
            gathered.view(ranks, logits.shape[0], logits.shape[1]),
            vocab_size=min(self.org_vocab_size, ranks * logits.shape[1]),
            backend="ixformer_native",
        )

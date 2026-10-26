"""Step5 native embedding/router integration without changing shared vLLM APIs."""

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

from vllm_iluvatar.ilu_util.arch import is_ivcore30


class Step5GateLinear(GateLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("_prepared_router_weight", None, persistent=False)
        self.register_buffer("_prepared_router_weight_nt", None, persistent=False)
        self._native_weights_ready = False

    def prepare_native_weights(self):
        if not self.is_unquantized or self.bias is not None:
            raise ValueError(
                "Step5 native router requires unquantized bias-free weights"
            )
        prepared_nt = self.weight.bfloat16().contiguous()
        if torch.equal(prepared_nt.float(), self.weight):
            self._prepared_router_weight = (
                prepared_nt.t().contiguous()
                if all(dim % 32 == 0 for dim in prepared_nt.shape)
                else None
            )
            self._prepared_router_weight_nt = (
                prepared_nt
                if prepared_nt.shape[1] == 4096 and prepared_nt.shape[0] % 16 == 0
                else None
            )
        else:
            self._prepared_router_weight = None
            self._prepared_router_weight_nt = None
        self._native_weights_ready = True

    def forward(self, x):
        if not x.is_cuda:
            return super().forward(x)
        if not self._native_weights_ready:
            raise RuntimeError("Step5 router weights must be prepared during loading")
        if (
            is_ivcore30()
            and x.ndim == 2
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.shape[0] <= 64
            and self._prepared_router_weight_nt is not None
        ):
            from ixformer.inference.functions import gemv_bf16_bf16_fp32

            output = gemv_bf16_bf16_fp32(
                x, self._prepared_router_weight_nt, backend="ixformer_native"
            )
            return self._return(output)

        if self._prepared_router_weight is not None:
            from ixformer.inference.functions import gemm_bf16_bf16_fp32

            output = gemm_bf16_bf16_fp32(
                x, self._prepared_router_weight, format="NN", backend="ixmm"
            )
        else:
            # Keep original FP32 weights when conversion would lose precision.
            output = torch.nn.functional.linear(x.float(), self.weight.float())
        return self._return(output)


class Step5VocabParallelEmbedding(VocabParallelEmbedding):
    def forward(self, input_ids):
        if not input_ids.is_cuda:
            return super().forward(input_ids)
        from ixformer.inference.functions.ops.miscs.vocab_parallel_embedding import (
            vocab_parallel_embedding,
        )

        shard = self.shard_indices
        local = vocab_parallel_embedding(
            input_ids.reshape(-1),
            self.weight,
            shard.org_vocab_start_index,
            shard.org_vocab_end_index,
            shard.num_org_vocab_padding,
            shard.added_vocab_start_index,
            shard.added_vocab_end_index,
            backend="ixformer_native",
        )
        local = local.view(*input_ids.shape, self.embedding_dim)
        if self.tp_size == 1:
            return local
        if self.parallel_group is not None:
            return self.parallel_group.all_reduce(local)
        return tensor_model_parallel_all_reduce(local)

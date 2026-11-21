# SPDX-License-Identifier: Apache-2.0
"""DFly2 integration for the vLLM 0.31 V2 runner."""

import torch
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from .dfly import DFly
from .dfly_inputs import prepare_dfly_inputs

logger = init_logger(__name__)


class FlashDFlySpeculator(DraftModelSpeculator):
    can_reuse_target_attn_metadata = True

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        if (
            self.num_speculative_steps != 5
            or self.speculative_config.draft_sample_method == "probabilistic"
        ):
            raise ValueError("HY4 Flash DFly2 expects five slots and greedy proposal")
        if self.speculative_config.enable_adaptive_verification:
            raise ValueError(
                "HY4 Flash fixed-depth verification requires adaptive verification off"
            )
        r = self.max_num_reqs
        self.hidden = torch.zeros((r, 3072), dtype=self.dtype, device=device)
        self.source = torch.zeros(r, dtype=torch.int64, device=device)
        self.anchors = torch.zeros_like(self.source)
        self.active = torch.zeros(r, dtype=torch.bool, device=device)
        self.query_ids = torch.zeros(r * 6, dtype=torch.int32, device=device)
        self.query_pos = torch.zeros(r * 6, dtype=torch.int64, device=device)
        self.padding = torch.ones(r * 6, dtype=torch.bool, device=device)
        self.starts = torch.zeros(r + 1, dtype=torch.int32, device=device)
        self.lengths = torch.zeros(r, dtype=torch.int32, device=device)
        self.maxlen = torch.tensor(self.max_model_len, dtype=torch.int32, device=device)
        self.mask_token_id = vllm_config.model_config.hf_config.dfly_mask_token_id
        self.graphs = {}
        self.use_graph = False
        self._capture_stream = None

    def load_draft_model(self, target_model, target_attn_layer_names):
        with torch.device(self.device):
            from vllm.utils.torch_utils import set_default_torch_dtype

            with set_default_torch_dtype(self.dtype):
                return DFly(self.vllm_config, target_model).eval()

    def set_attn(
        self,
        model_state,
        kv_cache_config,
        block_tables,
        target_input_buffers,
        target_attn_groups,
    ):
        self.model_state = model_state

    def init_cudagraph_manager(self, cudagraph_mode):
        self.use_graph = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL

    def _step(self, r):
        self.model(
            self.query_ids[: 6 * r],
            self.query_pos[: 6 * r],
            self.source[:r],
            self.active[:r],
            self.hidden[:r],
            self.draft_tokens[:r],
        )

    @torch.inference_mode()
    def capture(self):
        if self.use_graph:
            # Draft generation is local to the DP rank. Smaller buckets avoid
            # projecting, packing and sampling inactive queries on each step.
            sizes = {
                min(self.max_num_reqs, 1 << shift)
                for shift in range((self.max_num_reqs - 1).bit_length() + 1)
            }
            if self._capture_stream is None:
                self._capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream = self._capture_stream
            caller_stream = torch.cuda.current_stream(self.device)
            capture_stream.wait_stream(caller_stream)
            # cuBLASLt initializes workspaces per stream. Warming a different
            # stream can leave the first allocation inside capture, which
            # invalidates the graph on some CoreX driver versions.
            with torch.cuda.stream(capture_stream):
                self.active.zero_()
                for r in sorted(sizes, reverse=True):
                    self._step(r)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=capture_stream):
                        self._step(r)
                    self.graphs[r] = graph
            caller_stream.wait_stream(capture_stream)
            logger.info("HY4 DFly2 captured request sizes %s", sorted(self.graphs))

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
    ):
        batch = input_batch.num_reqs
        if dummy_run or not batch:
            return self.draft_tokens[:batch].fill_(-1)
        r = batch
        if self.graphs:
            r = min(self.max_num_reqs, 1 << (batch - 1).bit_length())
        prepare_dfly_inputs(
            input_ids=self.query_ids,
            positions=self.query_pos,
            is_padding=self.padding,
            query_start_loc=self.starts,
            seq_lens=self.lengths,
            active=self.active,
            source_rows=self.source,
            anchors=self.anchors,
            target_hidden_out=self.hidden,
            mrope_positions=None,
            draft_idx_mapping=self.idx_mapping,
            out_temperature=self.temperature,
            out_seeds=self.seeds,
            target_query_start_loc=input_batch.query_start_loc,
            target_positions=input_batch.positions,
            idx_mapping=input_batch.idx_mapping,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            target_hidden=last_hidden_states,
            mrope_delta=None,
            temperature=temperature,
            seeds=seeds,
            max_model_len=self.maxlen,
            mask_token_id=self.mask_token_id,
            query_width=6,
            num_reqs=batch,
            padded_batch=r,
        )
        if self.graphs:
            self.graphs[r].replay()
            logger.info_once("HY4 DFly2 replayed a real request batch in bucket %d", r)
        else:
            self._step(r)
        return self.draft_tokens[:batch]

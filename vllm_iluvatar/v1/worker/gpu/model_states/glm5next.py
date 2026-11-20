# SPDX-License-Identifier: Apache-2.0
"""GLM V2 state coordination: immediate fold before draft/next-step reuse."""

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState

from vllm_iluvatar.models.glm5next.recoverssm import get_recoverssm_resources
from vllm_iluvatar.v1.attention.backends.ilu_glm5next_kda import IluGlm5NextKDAMetadata


class IluGLMRecoverSSMState(RecoverSSMState):
    def __init__(self):
        super().__init__()

    def clear(self):
        self._step = None

    def record_step(self, attn_metadata, attn_groups, *, for_capture):
        self._step = None
        step = []
        seen = set()
        for group_list in attn_groups:
            for group in group_list:
                metadata = attn_metadata[group.layer_names[0]]
                if not isinstance(metadata, IluGlm5NextKDAMetadata):
                    continue
                resource_group = metadata.recoverssm_group
                if resource_group is None or metadata.num_spec_decodes == 0:
                    continue
                resources = resource_group.resources
                if for_capture:
                    resources.captured_generation = resources.generation
                    continue
                if resources.is_profiling or id(resource_group) in seen:
                    continue
                seen.add(id(resource_group))
                step.append(metadata)
        self._step = tuple(step) if step else None

    def commit_step(
        self,
        num_sampled,
        idx_mapping,
        *,
        state_indices,
        num_accepted_tokens,
    ):
        step, self._step = self._step, None
        if isinstance(num_sampled, int) or step is None:
            return
        for metadata in step:
            metadata.commit_recoverssm_state(num_sampled, idx_mapping=idx_mapping)
            metadata.recoverssm_group.finish(
                metadata.num_spec_decodes,
                state_indices,
                num_accepted_tokens,
            )


class IluGlm5NextModelState(MambaHybridModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if self.cache_config.use_kda_recoverssm:
            if get_recoverssm_resources(model) is None:
                raise RuntimeError(
                    "GLM RecoverSSM model resources were not initialized"
                )
            self.recoverssm = IluGLMRecoverSSMState()

    def add_request(self, req_index, new_req_data):
        super().add_request(req_index, new_req_data)
        if self.recoverssm is not None and self._align_mode:
            # Attention and Mamba block sizes can differ after hybrid planning.
            # Match the actual MambaSpec used by the planner and pre-copy.
            size = (
                self._mamba_spec.block_size
                if self._mamba_spec is not None
                else self.cache_config.mamba_block_size
            )
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // size
            )

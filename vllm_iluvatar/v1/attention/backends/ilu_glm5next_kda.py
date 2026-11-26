# SPDX-License-Identifier: Apache-2.0
"""GLM-specific immediate RecoverSSM metadata on the shared GDN interface."""

from dataclasses import dataclass, field

import torch
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)

from vllm_iluvatar.v1.attention.backends.ilu_gdn_attn import (
    IluGDNAttentionBackend,
    IluGDNAttentionMetadata,
    IluGDNAttentionMetadataBuilder,
)


@dataclass
class IluGlm5NextKDAMetadata(IluGDNAttentionMetadata, RecoverSSMMetadata):
    recoverssm_group: object | None = field(default=None, repr=False, compare=False)
    recoverssm_record_rows: torch.Tensor | None = None
    recoverssm_block_table: torch.Tensor | None = None
    recoverssm_block_size: int | None = None

    def commit_recoverssm_state(
        self,
        num_accepted_tokens: torch.Tensor,
        *,
        idx_mapping=None,
    ) -> RecoverSSMPostprocessMetadata | None:
        group = self.recoverssm_group
        if group is None or self.num_spec_decodes == 0:
            return None
        group.commit(
            num_accepted_tokens,
            self.num_spec_decodes,
            self.recoverssm_block_table,
            self.recoverssm_block_size,
            idx_mapping=idx_mapping,
        )
        if self.recoverssm_block_table is None:
            return None
        return RecoverSSMPostprocessMetadata(
            num_spec_decodes=self.num_spec_decodes,
            request_indices=group.request_rows[: self.num_spec_decodes],
            block_table=self.recoverssm_block_table,
            num_computed_tokens=group.computed_by_request,
            block_size=self.recoverssm_block_size,
        )


class IluGlm5NextKDAMetadataBuilder(IluGDNAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.ALWAYS
    mamba_aligned_state_indices: torch.Tensor | None = None

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        first = vllm_config.compilation_config.static_forward_context[layer_names[0]]
        resources = first._recoverssm_resources_ref()
        if resources is None:
            raise RuntimeError("GLM RecoverSSM storage owner is missing")
        self.group = resources.get_group(layer_names)
        self.supports_update_block_table = False
        # Keep only one checkpoint index per request. This is a view into memory
        # reserved during model loading, including padded metadata rows.
        self.spec_state_indices_tensor = self.group.source[:, None]
        self.spec_conv_state_indices = self.group.source
        if kv_cache_spec.num_speculative_blocks != 0:
            raise ValueError("GLM RecoverSSM must not reserve speculative state blocks")
        if kv_cache_spec.mamba_cache_mode == "align" and (
            kv_cache_spec.block_size < resources.spec_query_len
        ):
            raise ValueError("GLM RecoverSSM requires MambaSpec.block_size >= T")

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
    ):
        m = common_attn_metadata
        resources = self.group.resources
        if m.is_prefilling is None or m.is_prefilling.device.type != "cpu":
            raise ValueError("GLM RecoverSSM requires CPU request-phase metadata")
        lengths = m.query_start_loc_cpu.diff()
        # Overlong decode rows use the shared non-spec prefill path.
        active = (
            (~m.is_prefilling) & (lengths > 0) & (lengths <= resources.spec_query_len)
        )
        if int(active.sum()) > resources.record_capacity:
            raise ValueError("GLM RecoverSSM active requests exceed record capacity")
        pure_decode = not bool(((~active) & (lengths > 0)).any())
        align = self.kv_cache_spec.mamba_cache_mode == "align"

        if pure_decode:
            batch = m.num_reqs
            if align:
                if self.mamba_aligned_state_indices is None:
                    raise RuntimeError(
                        "GLM RecoverSSM align source slots were not prepared"
                    )
                source = self.mamba_aligned_state_indices[:batch, 0]
            else:
                source = m.block_table_tensor[:batch, 0]
            self.group.stage(source, m.query_start_loc, m.seq_lens, None, batch)
            return IluGlm5NextKDAMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                num_spec_decodes=batch,
                num_spec_decode_tokens=int(lengths.sum()),
                num_actual_tokens=m.num_actual_tokens,
                spec_query_start_loc=m.query_start_loc[: batch + 1],
                spec_state_indices_tensor=self.group.source[:batch, None],
                num_accepted_tokens=resources.neutral_accepted[:batch],
                recoverssm_group=self.group,
                recoverssm_record_rows=self.group.record_rows[:batch],
                recoverssm_block_table=m.block_table_tensor if align else None,
                recoverssm_block_size=self.kv_cache_spec.block_size if align else None,
            )

        # The shared builder normally drops zero-draft decodes. This routing mask
        # keeps all active decode rows (including L=1) on verify; actual lengths
        # always come from the GPU query starts, not these routing sentinels.
        routing = torch.full_like(lengths, -1, dtype=torch.int32)
        routing[active] = 1
        base = super().build(
            common_prefix_len,
            m,
            resources.neutral_accepted[: m.num_reqs],
            routing,
            fast_build,
        )
        batch = base.num_spec_decodes
        if batch == 0:
            return IluGlm5NextKDAMetadata(**vars(base))
        request_rows = active.nonzero(as_tuple=True)[0].to(dtype=torch.int32)
        self.group.stage(
            base.spec_state_indices_tensor[:batch, 0],
            base.spec_query_start_loc[: batch + 1],
            m.seq_lens,
            request_rows,
            batch,
        )
        base.spec_state_indices_tensor = self.group.source[:batch, None]
        return IluGlm5NextKDAMetadata(
            **vars(base),
            recoverssm_group=self.group,
            recoverssm_record_rows=self.group.record_rows[:batch],
            recoverssm_block_table=m.block_table_tensor if align else None,
            recoverssm_block_size=self.kv_cache_spec.block_size if align else None,
        )


class IluGlm5NextKDAAttentionBackend(IluGDNAttentionBackend):
    @staticmethod
    def get_builder_cls():
        return IluGlm5NextKDAMetadataBuilder

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True

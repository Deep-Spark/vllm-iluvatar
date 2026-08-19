# SPDX-License-Identifier: Apache-2.0
"""Iluvatar/corex adaptations for elastic-EP MoE routing.

Migrated from upstream ``vllm/model_executor/layers/fused_moe/iluvatar_runner/
ilu_elastic_ep.py`` + ``vllm/distributed/eplb/ilu_rebalance.py``.

Baseline : main_6e448d0

Contains:
  ilu_eplb_map_and_record       pure-torch EPLB mapping (replaces Triton kernel)
  ilu_rearrange_expert_weights  all_gather reshuffling (replaces ncclSend/Recv)
  ilu_is_monolithic             wrapped unquantized W16A16 detection
  ilu_kernel_handles_own_dispatch  dispatch ownership check
  ilu_modular_apply             W16A16 ixformer path after modular wrap
  install_ilu_elastic_ep_patches  entry point (called from IluMoERunner.__init__)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.distributed import ProcessGroup, all_gather

# floor(2^32 / phi): Knuth hash multiplier, same as the Triton kernel so the
# torch fallback picks the same replica / physical ids.
ILU_EPLB_KNUTH_MULTIPLIER = 2654435769


# -----------------------------------------------------------------------
# EPLB mapping (pure-torch replacement for the Triton kernel)
# -----------------------------------------------------------------------
def ilu_eplb_map_and_record(
    topk_ids: torch.Tensor,
    expert_load_view: torch.Tensor,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    record_enabled: torch.Tensor,
    num_unpadded_tokens: torch.Tensor | None = None,
    knuth_multiplier: int = ILU_EPLB_KNUTH_MULTIPLIER,
) -> torch.Tensor:
    """Pure-torch equivalent of ``_eplb_map_and_record_i32_kernel``."""
    numel = topk_ids.numel()
    if numel == 0:
        return topk_ids
    num_active_experts = topk_ids.shape[-1]
    num_logical_experts = logical_replica_count.shape[0]
    map_slots = logical_to_physical_map.shape[1]
    out_size = expert_load_view.shape[0]
    device = topk_ids.device

    flat = topk_ids.contiguous().reshape(-1).to(torch.int64)
    valid_expert = (flat >= 0) & (flat < num_logical_experts)
    safe_expert_id = torch.where(valid_expert, flat, torch.zeros_like(flat))

    replica_count = logical_replica_count.reshape(-1)[safe_expert_id].to(torch.int64)
    replica_count = torch.clamp(replica_count, min=1)
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    token_idx = idx // num_active_experts
    hashed = (token_idx * knuth_multiplier) & 0xFFFFFFFF
    replica_idx = hashed % replica_count

    map_index = safe_expert_id * map_slots + replica_idx
    physical_id = logical_to_physical_map.contiguous().reshape(-1)[map_index]
    physical_id = torch.where(
        valid_expert, physical_id, torch.full_like(physical_id, -1)
    )

    rec_mask = (physical_id >= 0) & (physical_id < out_size)
    if num_unpadded_tokens is not None:
        is_unpadded = idx < (
            num_unpadded_tokens.reshape(-1)[0].to(torch.int64) * num_active_experts
        )
        rec_mask &= is_unpadded
    rec_flag = record_enabled.reshape(-1)[0].to(expert_load_view.dtype)
    safe_phys = torch.where(rec_mask, physical_id, torch.zeros_like(physical_id)).to(
        torch.long
    )
    weights = rec_mask.to(expert_load_view.dtype) * rec_flag
    expert_load_view.index_add_(0, safe_phys, weights)

    return physical_id.to(topk_ids.dtype).reshape(topk_ids.shape)


# -----------------------------------------------------------------------
# Expert-weight reshuffling (all_gather replacement for ncclSend/Recv)
# -----------------------------------------------------------------------
def ilu_rearrange_expert_weights_allgather(
    num_moe_layers: int,
    num_local_physical_experts: int,
    ep_rank: int,
    ep_size: int,
    expert_weights: Sequence[Sequence[torch.Tensor]],
    old_global_expert_indices_cpu: np.ndarray,
    new_global_expert_indices_cpu: np.ndarray,
    ep_group: ProcessGroup,
) -> None:
    """All-gather based expert-weight reshuffle (deadlock-free on corex)."""
    for layer_idx in range(num_moe_layers):
        old_l = old_global_expert_indices_cpu[layer_idx]
        new_l = new_global_expert_indices_cpu[layer_idx]

        gathered: list[list[torch.Tensor]] = []
        for w in expert_weights[layer_idx]:
            w_contig = w.contiguous()
            buf = [torch.empty_like(w_contig) for _ in range(ep_size)]
            all_gather(buf, w_contig, group=ep_group)
            gathered.append(buf)

        for j in range(num_local_physical_experts):
            global_new = ep_rank * num_local_physical_experts + j
            logical = int(new_l[global_new])
            if logical < 0:
                continue
            matches = np.flatnonzero(old_l == logical)
            if matches.size == 0:
                continue

            local_lo = ep_rank * num_local_physical_experts
            local_hi = local_lo + num_local_physical_experts
            local_matches = matches[(matches >= local_lo) & (matches < local_hi)]
            src_g = int(local_matches[0] if local_matches.size else matches[0])
            src_r, src_i = divmod(src_g, num_local_physical_experts)
            for wi in range(len(expert_weights[layer_idx])):
                expert_weights[layer_idx][wi][j].copy_(gathered[wi][src_r][src_i])

        del gathered
    torch.cuda.synchronize()


# -----------------------------------------------------------------------
# Modular W16A16 helpers
# -----------------------------------------------------------------------
def ilu_is_monolithic(modular_method) -> bool:
    """Whether a ``FusedMoEModularMethod`` wraps the W16A16 unquantized path.

    Elastic EP may wrap ``UnquantizedFusedMoEMethod`` into
    ``FusedMoEModularMethod``. That wrapper's community apply path would call
    ``moe_kernel.apply``; for Iluvatar W16A16 we instead keep using the plugin's
    ixformer path.
    """
    cached = getattr(modular_method, "_ilu_is_monolithic", None)
    if cached is None:
        quant_method = getattr(modular_method, "old_quant_method", modular_method)
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )

        cached = isinstance(quant_method, UnquantizedFusedMoEMethod)
        modular_method._ilu_is_monolithic = cached
    return cached


def ilu_kernel_handles_own_dispatch(quant_method) -> bool:
    """True when the modular kernel does its own DP/EP dispatch/combine,
    so the runner skips its own. Monolithic Iluvatar layers still need
    the runner's dispatch."""
    from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
        FusedMoEModularMethod,
    )

    if not isinstance(quant_method, FusedMoEModularMethod):
        return False
    return not ilu_is_monolithic(quant_method)


def ilu_modular_apply(
    modular_method,
    layer,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor | None:
    """Run the plugin-local W16A16 path after elastic EP modular wrapping.

    Returns the fused output for wrapped unquantized layers, else None so the
    caller takes the regular modular path.
    """
    if not ilu_is_monolithic(modular_method):
        return None

    # Fork delta formerly patched community fused_moe.py. Keep this dispatch in
    # the plugin so community fused_moe.py stays identical to upstream main.
    from vllm_iluvatar.model_executor.layers.fused_moe.ilu_fused_moe import (
        ilu_w16a16_fused_experts,
    )

    return ilu_w16a16_fused_experts(
        hidden_states=x,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=layer.activation,
        apply_router_weight_on_input=layer.apply_router_weight_on_input,
        global_num_experts=layer.global_num_experts,
        expert_map=layer.expert_map,
        quant_config=modular_method.moe_quant_config,
    )


# -----------------------------------------------------------------------
# Patch installers
# -----------------------------------------------------------------------
def _install_rebalance_patches() -> None:
    """Patch ``rearrange_expert_weights_inplace`` to the all_gather
    reshuffle. Idempotent."""
    from vllm.distributed.eplb import eplb_state, rebalance_execute

    if getattr(rebalance_execute, "_ilu_elastic_ep_patched", False):
        return

    _orig = rebalance_execute.rearrange_expert_weights_inplace

    def rearrange_expert_weights_inplace(
        old_global_expert_indices,
        new_global_expert_indices,
        expert_weights,
        expert_buffer,
        ep_group,
        communicator,
        is_profile: bool = False,
        rank_mapping: dict[int, int] | None = None,
    ) -> None:
        if is_profile:
            return _orig(
                old_global_expert_indices,
                new_global_expert_indices,
                expert_weights,
                expert_buffer,
                ep_group,
                communicator,
                is_profile=is_profile,
                rank_mapping=rank_mapping,
            )

        if rank_mapping is not None:
            if len(rank_mapping) == ep_group.size():
                new_global_expert_indices = (
                    rebalance_execute._map_new_expert_indices_with_rank_mapping(
                        new_global_expert_indices, rank_mapping
                    )
                )
            else:
                old_global_expert_indices = (
                    rebalance_execute._map_old_expert_indices_with_rank_mapping(
                        old_global_expert_indices,
                        rank_mapping,
                        ep_group.size(),
                    )
                )

        assert old_global_expert_indices.shape[1] == new_global_expert_indices.shape[1]
        num_moe_layers = old_global_expert_indices.shape[0]
        num_local_physical_experts = expert_weights[0][0].shape[0]
        ep_size = ep_group.size()
        ep_rank = ep_group.rank()

        ilu_rearrange_expert_weights_allgather(
            num_moe_layers=num_moe_layers,
            num_local_physical_experts=num_local_physical_experts,
            ep_rank=ep_rank,
            ep_size=ep_size,
            expert_weights=expert_weights,
            old_global_expert_indices_cpu=(old_global_expert_indices.cpu().numpy()),
            new_global_expert_indices_cpu=(new_global_expert_indices.cpu().numpy()),
            ep_group=ep_group,
        )

    rebalance_execute.rearrange_expert_weights_inplace = (
        rearrange_expert_weights_inplace
    )
    eplb_state.rearrange_expert_weights_inplace = rearrange_expert_weights_inplace
    rebalance_execute._ilu_elastic_ep_patched = True


def _install_eplb_map_patches() -> None:
    """Patch ``base_router.eplb_map_to_physical_and_record`` to the
    pure-torch mapping. Idempotent."""
    from vllm.model_executor.layers.fused_moe.router import base_router

    if getattr(base_router, "_ilu_elastic_ep_patched", False):
        return

    base_router.eplb_map_to_physical_and_record = ilu_eplb_map_and_record
    base_router._ilu_elastic_ep_patched = True


def _install_modular_method_patches() -> None:
    """Patch ``FusedMoEModularMethod.apply`` for wrapped W16A16 layers.

    The W16A16 path no longer lives in community modular kernels. This wrapper
    intercepts only modular wrappers around ``UnquantizedFusedMoEMethod`` and
    lets every other modular quant method fall through unchanged.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
        FusedMoEModularMethod,
    )

    if getattr(FusedMoEModularMethod, "_ilu_elastic_ep_patched", False):
        return

    _orig_apply = FusedMoEModularMethod.apply

    def apply(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts,
        shared_experts_input,
    ):
        ilu_output = ilu_modular_apply(self, layer, x, topk_weights, topk_ids)
        if ilu_output is not None:
            if shared_experts_input is not None:
                return ilu_output + shared_experts_input
            return ilu_output
        return _orig_apply(
            self,
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts,
            shared_experts_input,
        )

    FusedMoEModularMethod.apply = apply
    FusedMoEModularMethod._ilu_elastic_ep_patched = True


_PATCHES_INSTALLED = False


def install_ilu_elastic_ep_patches() -> None:
    """Install all corex/Iluvatar elastic-EP runtime patches.

    Called from ``IluMoERunner.__init__`` (model build time), before any
    patched path runs. Idempotent."""
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return

    _install_rebalance_patches()
    _install_eplb_map_patches()
    _install_modular_method_patches()
    _PATCHES_INSTALLED = True

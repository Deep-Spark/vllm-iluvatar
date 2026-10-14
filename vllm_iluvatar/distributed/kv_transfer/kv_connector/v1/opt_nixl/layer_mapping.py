"""Pair local and remote registered pointers layer by layer for a PP slice.

A transfer to one remote rank carries every KV tensor of the overlap at once,
whatever group it belongs to: each pointer pair keeps its own local and remote
group id, so the writer indexes each side's block-id tuple with that side's
own group numbering.  The two sides may number and split their groups
differently (P holds forwarder layers that D does not have).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheSpec

    from .metadata import NixlAgentMetadata

# (offset within one block, payload bytes)
Descriptor = tuple[int, int]
# Per registered pointer of a layer, its descriptor segments.
LayerDescriptorMapping = list[list[Descriptor]]


def _region_sort_key(layer_name: str) -> tuple[tuple[int, int | str], ...]:
    # Copied from upstream nixl/base_worker.py so that "layers.10" sorts
    # after "layers.9" and sibling tensors of one layer keep a stable order.
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", layer_name)
    )


@dataclass(frozen=True)
class RegistrationView:
    """The per-pointer registration tables one side advertises."""

    ptr: dict[str, list[int]]
    block_stride: dict[str, list[int]]
    desc: dict[str, LayerDescriptorMapping]
    num_blocks: dict[str, list[int]]
    group: dict[str, list[int]]
    kernel_blocks: dict[str, list[int]]
    block_tokens: dict[str, list[int]]
    backend: dict[str, str]

    @classmethod
    def from_agent_metadata(cls, metadata: NixlAgentMetadata) -> RegistrationView:
        return cls(
            ptr=metadata.layer_ptr_mapping,
            block_stride=metadata.layer_block_len_mapping,
            desc=metadata.layer_desc_mapping,
            num_blocks=metadata.layer_num_blocks_mapping,
            group=metadata.layer_group_mapping,
            kernel_blocks=metadata.layer_kernel_blocks_mapping,
            block_tokens=metadata.layer_block_tokens_mapping,
            backend=metadata.layer_backend_mapping,
        )


@dataclass(frozen=True)
class PointerPair:
    """One registered local pointer and the remote pointer it is written to."""

    kv_name: str
    spec: KVCacheSpec
    local_ptr: int
    remote_ptr: int
    local_desc: list[Descriptor]
    remote_desc: list[Descriptor]
    local_block_stride: int
    remote_block_stride: int
    local_num_blocks: int
    remote_num_blocks: int
    local_group_id: int
    remote_group_id: int
    kernel_blocks: int


@dataclass(frozen=True)
class LayerSelection:
    """All pointer pairs of one model layer, kv names in sorted order."""

    # Global model layer index.
    layer_index: int
    pointers: tuple[PointerPair, ...]


class LayerMappingPlanner:
    """Walk the local layers of a PP overlap in order and pair their pointers."""

    def __init__(
        self,
        *,
        local: RegistrationView,
        layer_specs: dict[str, KVCacheSpec],
        layer_global_index: dict[str, int],
        pp_layer_start: int,
        num_layers: int,
        local_tp_size: int,
        has_mamba: bool,
    ) -> None:
        self.local = local
        self.layer_specs = layer_specs
        self.layer_global_index = layer_global_index
        self.pp_layer_start = pp_layer_start
        self.num_layers = num_layers
        self.local_tp_size = local_tp_size
        self.has_mamba = has_mamba

    def select(
        self,
        *,
        local_start: int,
        local_end: int,
        remote: RegistrationView,
        remote_tp_size: int,
    ) -> list[LayerSelection]:
        """Pair every locally registered kv tensor in PP-relative [start, end).

        Layers the model left unregistered (``skip_layers``) are absent from
        ``local.ptr`` and simply do not appear.  A layer registered here but
        missing on the peer is a configuration error.  Block ids travel
        unmodified, so both pointers of a pair must split a block into the
        same number of kernel blocks and use the same block size in tokens.
        """

        if not 0 <= local_start < local_end <= self.num_layers:
            raise ValueError(f"invalid local layer slice [{local_start}, {local_end})")

        names: list[tuple[int, str]] = []
        for kv_name in self.local.ptr:
            if "mtp" in kv_name:
                continue
            global_index = self.layer_global_index[kv_name]
            if local_start <= global_index - self.pp_layer_start < local_end:
                names.append((global_index, kv_name))
        names.sort(key=lambda item: (item[0], _region_sort_key(item[1])))

        layers: list[LayerSelection] = []
        pending: list[PointerPair] = []
        pending_index: int | None = None
        for global_index, kv_name in names:
            if pending_index is not None and global_index != pending_index:
                layers.append(LayerSelection(pending_index, tuple(pending)))
                pending = []
            pending_index = global_index
            pending.extend(self._pair(kv_name, remote, remote_tp_size))
        if pending_index is not None:
            layers.append(LayerSelection(pending_index, tuple(pending)))
        return layers

    def _pair(
        self,
        kv_name: str,
        remote: RegistrationView,
        remote_tp_size: int,
    ) -> list[PointerPair]:
        if kv_name not in remote.ptr:
            raise RuntimeError(
                f"remote handshake is missing layer {kv_name}; it is registered "
                "locally, so either the peer lists it in skip_layers and this "
                "side did not, or the two sides run different models"
            )
        local_backend = self.local.backend.get(kv_name)
        remote_backend = remote.backend.get(kv_name)
        if local_backend != remote_backend:
            raise ValueError(
                f"{kv_name}: attention backend differs (local {local_backend}, "
                f"remote {remote_backend}); both sides must lay this kv out "
                "with the same backend"
            )
        local_ptrs = self.local.ptr[kv_name]
        remote_ptrs = remote.ptr[kv_name]
        if len(local_ptrs) != len(remote_ptrs):
            raise ValueError(
                f"{kv_name}: local/remote pointer count mismatch "
                f"({len(local_ptrs)} != {len(remote_ptrs)})"
            )
        spec = self.layer_specs[kv_name]
        split_mamba = self.has_mamba and (
            isinstance(spec, MambaSpec) or not isinstance(spec, MLAAttentionSpec)
        )
        if split_mamba and len(local_ptrs) != 1:
            raise ValueError(
                f"{kv_name}: hybrid Mamba transfer expects one registered "
                "pointer per layer"
            )

        pairs: list[PointerPair] = []
        for index, (local_ptr, remote_ptr) in enumerate(zip(local_ptrs, remote_ptrs)):
            local_kernel = self.local.kernel_blocks[kv_name][index]
            remote_kernel = remote.kernel_blocks[kv_name][index]
            if local_kernel != remote_kernel:
                raise ValueError(
                    f"{kv_name}[{index}]: kernel blocks per block differ "
                    f"(local {local_kernel}, remote {remote_kernel}); both "
                    "sides must use the same kernel block size for this kv"
                )
            local_tokens = self.local.block_tokens[kv_name][index]
            remote_tokens = remote.block_tokens[kv_name][index]
            if local_tokens != remote_tokens:
                raise ValueError(
                    f"{kv_name}[{index}]: block size differs (local "
                    f"{local_tokens}, remote {remote_tokens} tokens); both "
                    "sides must use the same block size for this kv"
                )

            local_desc = list(self.local.desc[kv_name][index])
            remote_desc = list(remote.desc[kv_name][index])
            # Registration keeps both the split segments (conv/ssm or K/V
            # halves) and a trailing whole-block segment: same TP moves the
            # whole block, heterogeneous TP needs the split segments.
            if split_mamba:
                if self.local_tp_size == remote_tp_size:
                    local_desc, remote_desc = local_desc[-1:], remote_desc[-1:]
                else:
                    local_desc, remote_desc = local_desc[:-1], remote_desc[:-1]
            if len(local_desc) != len(remote_desc):
                raise ValueError(
                    f"{kv_name}[{index}]: local/remote segment count mismatch "
                    f"({len(local_desc)} != {len(remote_desc)})"
                )

            pairs.append(
                PointerPair(
                    kv_name=kv_name,
                    spec=spec,
                    local_ptr=local_ptr,
                    remote_ptr=remote_ptr,
                    local_desc=local_desc,
                    remote_desc=remote_desc,
                    local_block_stride=self.local.block_stride[kv_name][index],
                    remote_block_stride=remote.block_stride[kv_name][index],
                    local_num_blocks=self.local.num_blocks[kv_name][index],
                    remote_num_blocks=remote.num_blocks[kv_name][index],
                    local_group_id=self.local.group[kv_name][index],
                    remote_group_id=remote.group[kv_name][index],
                    kernel_blocks=local_kernel,
                )
            )
        return pairs

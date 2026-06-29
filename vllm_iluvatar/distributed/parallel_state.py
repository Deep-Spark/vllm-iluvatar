# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/distributed/parallel_state.py

ILU override implementations for GroupCoordinator.{broadcast,
broadcast_tensor_dict}. This module holds ONLY the override implementations;
the wiring (register/_apply, fingerprint + signature guards, global binding)
lives in ``ilu_patch.general_patches.patch_parallel_state`` and is
auto-discovered during ``register_general``.

Target   : vllm.distributed.parallel_state.GroupCoordinator.{broadcast,
           broadcast_tensor_dict}
Baseline : main_e0871ad2259
Why      : route broadcast / GPU-tensor broadcast through the device
           communicator so Iluvatar (ixformer) collectives are used instead of
           torch.distributed defaults.
How      : broadcast            -> wrapper delegating to the community method;
           broadcast_tensor_dict -> full-method copy (delta in MODIFICATION
                                    blocks below).
Scope    : all TP/PP runs.
Remove   : when upstream GroupCoordinator broadcasts via device_communicator
           natively.

The copied ``broadcast_tensor_dict`` references community globals (torch,
TensorMetadata, _split_tensor_dict). They are placeholders here and bound onto
this module by ``patch_parallel_state._apply()`` at register-time (kept
lazy so importing this module never pulls in vllm/torch).
"""

from __future__ import annotations

# Community globals used by the copied broadcast_tensor_dict body; bound by
# patch_parallel_state._apply() (kept lazy so importing this module never
# pulls in vllm/torch).
torch = None
TensorMetadata = None
_split_tensor_dict = None


# [ILU override] GroupCoordinator.broadcast — delta: prefer
# device_communicator.broadcast when world_size > 1; otherwise delegate to the
# community implementation unchanged.
def _make_broadcast(orig):
    def broadcast(self, input_, src=0):
        if self.world_size != 1 and self.device_communicator is not None:
            return self.device_communicator.broadcast(input_, src)
        return orig(self, input_, src)

    return broadcast


# [ILU override] GroupCoordinator.broadcast_tensor_dict — delta: GPU tensors are
# broadcast through device_communicator.broadcast_device_tensor when available,
# and None async handles are tolerated. Full-method copy of the baseline; only
# the lines inside the MODIFICATION blocks differ from community.
def broadcast_tensor_dict(
    self,
    tensor_dict=None,
    src=0,
    group=None,
    metadata_group=None,
):
    """Broadcast the input tensor dictionary.
    NOTE: `src` is the local rank of the source rank.
    """
    # Bypass the function if we are using only 1 GPU.
    if not torch.distributed.is_initialized() or self.world_size == 1:
        return tensor_dict

    group = self.device_group
    metadata_group = self.cpu_group
    assert src < self.world_size, f"Invalid src rank ({src})"

    rank_in_group = self.rank_in_group
    if rank_in_group == src:
        metadata_list = []
        assert isinstance(tensor_dict, dict), (
            f"Expecting a dictionary, got {type(tensor_dict)}"
        )
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # `metadata_list` lives in CPU memory.
        # `broadcast_object_list` has serialization & deserialization,
        # all happening on CPU. Therefore, we can use the CPU group.
        self.broadcast_object(metadata_list, src=src)
        async_handles = []
        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip broadcasting empty tensors.
                continue
            if tensor.is_cpu:
                # use metadata_group for CPU tensors
                handle = torch.distributed.broadcast(
                    tensor, src=self.ranks[src], group=metadata_group, async_op=True
                )
            else:
                # use group for GPU tensors
                # >>> ILU MODIFICATION >>>
                dc = self.device_communicator
                if dc is not None and hasattr(dc, "broadcast_device_tensor"):
                    handle = dc.broadcast_device_tensor(
                        tensor, self.ranks[src], group=group
                    )
                else:
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=group, async_op=True
                    )
                # <<< ILU MODIFICATION <<<
            async_handles.append(handle)
        for async_handle in async_handles:
            # >>> ILU MODIFICATION >>>
            if async_handle is not None:
                async_handle.wait()
            # <<< ILU MODIFICATION <<<

    else:
        metadata_list = self.broadcast_object(None, src=src)
        tensor_dict = {}
        async_handles = []
        for key, value in metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    tensor_dict[key] = tensor
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(
                        tensor,
                        src=self.ranks[src],
                        group=metadata_group,
                        async_op=True,
                    )
                else:
                    # use group for GPU tensors
                    # >>> ILU MODIFICATION >>>
                    dc = self.device_communicator
                    if dc is not None and hasattr(dc, "broadcast_device_tensor"):
                        handle = dc.broadcast_device_tensor(
                            tensor, self.ranks[src], group=group
                        )
                    else:
                        handle = torch.distributed.broadcast(
                            tensor, src=self.ranks[src], group=group, async_op=True
                        )
                    # <<< ILU MODIFICATION <<<
                async_handles.append(handle)
                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        for async_handle in async_handles:
            # >>> ILU MODIFICATION >>>
            if async_handle is not None:
                async_handle.wait()
            # <<< ILU MODIFICATION <<<
    return tensor_dict

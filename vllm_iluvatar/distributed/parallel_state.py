# SPDX-License-Identifier: Apache-2.0
"""Corresponds to: vllm/distributed/parallel_state.py

ILU override implementations for GroupCoordinator.{broadcast,
broadcast_tensor_dict, isend_tensor_dict, irecv_tensor_dict}.
This module holds ONLY the override implementations; the wiring
(register/_apply, fingerprint + signature guards, global binding) lives in
``ilu_patch.general_patches.patch_parallel_state`` and is auto-discovered
during ``register_general``.

Target   : vllm.distributed.parallel_state.GroupCoordinator.{broadcast,
           broadcast_tensor_dict, isend_tensor_dict, irecv_tensor_dict}
Baseline : main_88e5e2c57be8
Why      : 1) route broadcast / GPU-tensor broadcast through the device
              communicator so Iluvatar (ixformer) collectives are used instead
              of torch.distributed defaults.
           2) on ivcore30, torch.distributed.isend/irecv are not reliable;
              replace with blocking send/recv for PP tensor exchange.
How      : broadcast            -> wrapper delegating to the community method;
           broadcast_tensor_dict -> full-method copy (delta in MODIFICATION
                                    blocks below);
           isend_tensor_dict    -> wrapper: ivcore30 uses full-method copy
                                    with send (blocking), ivcore11 delegates
                                    to community;
           irecv_tensor_dict    -> wrapper: ivcore30 uses full-method copy
                                    with recv (blocking), ivcore11 delegates
                                    to community.
Scope    : all TP/PP runs.
Remove   : broadcast* — when upstream GroupCoordinator broadcasts via
           device_communicator natively.
           isend/irecv — when ivcore30 supports async isend/irecv or upstream
           adds a configurable sync/async mode.

The copied methods reference community globals (torch, TensorMetadata,
_split_tensor_dict, Handle, Callable). They are placeholders here and bound
onto this module by ``patch_parallel_state._apply()`` at register-time (kept
lazy so importing this module never pulls in vllm/torch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

# Community globals used by the copied method bodies; bound by
# patch_parallel_state._apply() (kept lazy so importing this module never
# pulls in vllm/torch).
torch = None
TensorMetadata = None
_split_tensor_dict = None
Handle = None
Callable = None


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
    tensor_dict: dict[str, torch.Tensor | Any] | None = None,
    src: int = 0,
    group: ProcessGroup | None = None,
    metadata_group: ProcessGroup | None = None,
) -> dict[str, torch.Tensor | Any] | None:
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
        metadata_list: list[tuple[Any, Any]] = []
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


# ---------------------------------------------------------------------------
# ivcore30: blocking send/recv for PP tensor exchange
# ---------------------------------------------------------------------------

from vllm_iluvatar.ilu_util.arch import is_ivcore30 as _is_ivcore30_fn


# [ILU override] GroupCoordinator.isend_tensor_dict — delta: on ivcore30,
# use blocking torch.distributed.send instead of isend and return empty
# handles.  Full-method copy of the baseline; only the lines inside the
# MODIFICATION blocks differ from community.
def _ilu_isend_tensor_dict_sync(
    self,
    tensor_dict: dict[str, "torch.Tensor | Any"],
    dst: int | None = None,
    all_gather_group: "GroupCoordinator | None" = None,
    all_gather_tensors: dict[str, bool] | None = None,
) -> "list[Handle]":
    if self.world_size <= 1:
        return []

    if dst is None:
        dst = (self.rank_in_group + 1) % self.world_size
    assert dst < self.world_size, f"Invalid dst rank ({dst})"

    if self.use_cpu_custom_send_recv:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        self.device_communicator.send_tensor_dict(  # type: ignore
            tensor_dict, dst
        )
        return []

    all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
    all_gather_rank = (
        0 if all_gather_group is None else all_gather_group.rank_in_group
    )

    group = self.device_group
    metadata_group = self.cpu_group

    metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
    self.send_object(metadata_list, dst=dst)

    tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
    assert len(tensor_keys) == len(tensor_list)

    handles: list[Handle] = []
    for key, tensor in zip(tensor_keys, tensor_list):
        if tensor.numel() == 0:
            continue

        if self._should_use_all_gather(
            key, tensor.numel(), all_gather_group, all_gather_tensors
        ):
            tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

        comm_group = metadata_group if tensor.is_cpu else group
        # >>> ILU MODIFICATION >>>
        torch.distributed.send(
            tensor, dst=self.ranks[dst], group=comm_group
        )
        # <<< ILU MODIFICATION <<<
        if tensor.is_cuda:
            tensor.record_stream(torch.cuda.current_stream(tensor.device))

    return handles


# [ILU override] GroupCoordinator.irecv_tensor_dict — delta: on ivcore30,
# use blocking torch.distributed.recv instead of irecv and return empty
# handles.  Full-method copy of the baseline; only the lines inside the
# MODIFICATION blocks differ from community.
def _ilu_irecv_tensor_dict_sync(
    self,
    src: int | None = None,
    all_gather_group: "GroupCoordinator | None" = None,
    all_gather_tensors: dict[str, bool] | None = None,
) -> "tuple[dict[str, torch.Tensor | Any] | None, list[Handle], list[Callable[[], None]]]":
    if not torch.distributed.is_initialized() or self.world_size == 1:
        return None, [], []

    if src is None:
        src = (self.rank_in_group - 1) % self.world_size
    assert src < self.world_size, f"Invalid src rank ({src})"

    if self.use_cpu_custom_send_recv:
        if self.device_communicator is None:
            raise ValueError("No device communicator found")
        sync_tensor_dict = self.device_communicator.recv_tensor_dict(  # type: ignore
            src
        )
        return sync_tensor_dict, [], []

    all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
    all_gather_rank = (
        0 if all_gather_group is None else all_gather_group.rank_in_group
    )

    group = self.device_group
    metadata_group = self.cpu_group

    recv_metadata_list = self.recv_object(src=src)
    tensor_dict: dict[str, Any] = {}
    handles: list[Handle] = []
    postprocess: list[Callable[[], None]] = []

    for key, value in recv_metadata_list:
        if isinstance(value, TensorMetadata):
            full_tensor = torch.empty(
                value.size, dtype=value.dtype, device=value.device
            )
            if full_tensor.numel() == 0:
                tensor_dict[key] = full_tensor
                continue

            if self._should_use_all_gather(
                key, full_tensor.numel(), all_gather_group, all_gather_tensors
            ):
                orig_shape = full_tensor.shape
                slice_tensor = full_tensor.reshape(all_gather_size, -1)[
                    all_gather_rank
                ]
                comm_group = metadata_group if slice_tensor.is_cpu else group
                # >>> ILU MODIFICATION >>>
                torch.distributed.recv(
                    slice_tensor, src=self.ranks[src], group=comm_group
                )
                # <<< ILU MODIFICATION <<<

                def _postprocess(
                    key: str = key,
                    slice_tensor: torch.Tensor = slice_tensor,
                    orig_shape: tuple[int, ...] = tuple(orig_shape),
                    all_gather_group=all_gather_group,
                ) -> None:
                    assert all_gather_group is not None
                    tensor_dict[key] = all_gather_group.all_gather(
                        slice_tensor, dim=0
                    ).reshape(orig_shape)

                postprocess.append(_postprocess)
                tensor_dict[key] = slice_tensor
            else:
                comm_group = metadata_group if full_tensor.is_cpu else group
                # >>> ILU MODIFICATION >>>
                torch.distributed.recv(
                    full_tensor, src=self.ranks[src], group=comm_group
                )
                # <<< ILU MODIFICATION <<<
                tensor_dict[key] = full_tensor
        else:
            tensor_dict[key] = value

    return tensor_dict, handles, postprocess


def _make_isend_tensor_dict(orig):
    """Wrapper: ivcore30 → blocking send; ivcore11 → community isend."""
    def isend_tensor_dict(self, tensor_dict, dst=None,
                          all_gather_group=None, all_gather_tensors=None):
        if _is_ivcore30_fn():
            return _ilu_isend_tensor_dict_sync(
                self, tensor_dict, dst, all_gather_group, all_gather_tensors
            )
        return orig(self, tensor_dict, dst, all_gather_group, all_gather_tensors)
    return isend_tensor_dict


def _make_irecv_tensor_dict(orig):
    """Wrapper: ivcore30 → blocking recv; ivcore11 → community irecv."""
    def irecv_tensor_dict(self, src=None,
                          all_gather_group=None, all_gather_tensors=None):
        if _is_ivcore30_fn():
            return _ilu_irecv_tensor_dict_sync(
                self, src, all_gather_group, all_gather_tensors
            )
        return orig(self, src, all_gather_group, all_gather_tensors)
    return irecv_tensor_dict

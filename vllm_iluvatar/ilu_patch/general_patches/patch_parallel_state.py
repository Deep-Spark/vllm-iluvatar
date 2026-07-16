# SPDX-License-Identifier: Apache-2.0
"""Wiring for the parallel_state ILU override.

Corresponds to: vllm/distributed/parallel_state.py

ILU general-plugins patch, auto-discovered from the centralized
``ilu_patch.general_patches`` package during ``register_general``.

The override implementations live in ``vllm_iluvatar.distributed.parallel_state``;
this module only does register-time wiring: fingerprint + signature guards,
binding the community globals onto the impl module, and installing the
overrides onto GroupCoordinator.

Target   : vllm.distributed.parallel_state.GroupCoordinator.{broadcast,
           broadcast_tensor_dict, isend_tensor_dict, irecv_tensor_dict}
Baseline : main_88e5e2c57be8
How      : broadcast            -> wrapper delegating to the community method;
           broadcast_tensor_dict -> full-method copy (in parallel_state.py),
                                    guarded by a source fingerprint against the
                                    baseline snapshot inlined below;
           isend_tensor_dict    -> wrapper: ivcore30 uses full-method copy
                                    with blocking send, ivcore11 delegates to
                                    community (fingerprint-guarded);
           irecv_tensor_dict    -> wrapper: ivcore30 uses full-method copy
                                    with blocking recv, ivcore11 delegates to
                                    community (fingerprint-guarded).
Fingerprint: broadcast_tensor_dict, isend_tensor_dict, irecv_tensor_dict
           community sources are hashed at register-time against the baseline
           snapshots inlined below; gated by VLLM_ILU_PATCH_CHECK_FINGERPRINT
           (default on) so startup fails fast if upstream changed the body.
"""

from __future__ import annotations

import inspect

from vllm_iluvatar.ilu_patch import general_patch as p

# Verbatim community-baseline source of GroupCoordinator.broadcast_tensor_dict
# (4-space method indent, trailing newline), used as the drift anchor by the
# fingerprint guard. Do NOT hand-edit; when re-syncing to a new vLLM baseline,
# replace this with the new community source AND update the copy in
# parallel_state.py together. Triple-SINGLE quotes are required because the
# method body contains a triple-double-quoted docstring.
#
# Baseline: main_88e5e2c57be8
BROADCAST_TENSOR_DICT = '''    def broadcast_tensor_dict(
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
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=group, async_op=True
                    )
                async_handles.append(handle)
            for async_handle in async_handles:
                async_handle.wait()

        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(
                        value.size, dtype=value.dtype, device=value.device
                    )
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
                        handle = torch.distributed.broadcast(
                            tensor, src=self.ranks[src], group=group, async_op=True
                        )
                    async_handles.append(handle)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict
'''

# Verbatim community-baseline source of GroupCoordinator.isend_tensor_dict
# (4-space method indent, trailing newline), used as the drift anchor.
# Baseline: main_88e5e2c57be8
ISEND_TENSOR_DICT = '''    def isend_tensor_dict(
        self,
        tensor_dict: dict[str, torch.Tensor | Any],
        dst: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> list[Handle]:
        if self.world_size <= 1:
            return []

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            # custom device communicator path is synchronous
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
            handle = torch.distributed.isend(
                tensor, dst=self.ranks[dst], group=comm_group
            )
            if tensor.is_cuda:
                tensor.record_stream(torch.cuda.current_stream(tensor.device))
            handles.append(handle)

        return handles
'''

# Verbatim community-baseline source of GroupCoordinator.irecv_tensor_dict
# (4-space method indent, trailing newline), used as the drift anchor.
# Baseline: main_88e5e2c57be8
IRECV_TENSOR_DICT = '''    def irecv_tensor_dict(
        self,
        src: int | None = None,
        all_gather_group: "GroupCoordinator | None" = None,
        all_gather_tensors: dict[str, bool] | None = None,
    ) -> tuple[
        dict[str, torch.Tensor | Any] | None,
        list[Handle],
        list[Callable[[], None]],
    ]:
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None, [], []

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"

        if self.use_cpu_custom_send_recv:
            if self.device_communicator is None:
                raise ValueError("No device communicator found")
            # custom device communicator path is synchronous
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
                    handle = torch.distributed.irecv(
                        slice_tensor, src=self.ranks[src], group=comm_group
                    )
                    handles.append(handle)

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
                    handle = torch.distributed.irecv(
                        full_tensor, src=self.ranks[src], group=comm_group
                    )
                    handles.append(handle)
                    tensor_dict[key] = full_tensor
            else:
                tensor_dict[key] = value

        return tensor_dict, handles, postprocess
'''


def register() -> None:
    """Register ILU GroupCoordinator overrides."""
    p.register_once(
        __name__,
        _apply,
        detail="broadcast wrapper, broadcast_tensor_dict copy, "
               "isend/irecv_tensor_dict ivcore30 sync wrapper",
    )


def _apply() -> None:
    import collections.abc as _abc
    import torch as _torch
    import vllm.distributed.parallel_state as _m

    import vllm_iluvatar.distributed.parallel_state as impl

    gc = _m.GroupCoordinator

    # --- broadcast_tensor_dict fingerprint + signature ---
    p.check_fingerprint(
        "GroupCoordinator.broadcast_tensor_dict",
        inspect.getsource(gc.broadcast_tensor_dict),
        BROADCAST_TENSOR_DICT,
        resync_hint=(
            "Re-sync vllm_iluvatar/distributed/parallel_state.py (the copy) and "
            "the BROADCAST_TENSOR_DICT baseline in "
            "ilu_patch/general_patches/patch_parallel_state.py to the new "
            "community source of GroupCoordinator.broadcast_tensor_dict."
        ),
    )
    p.check_signatures(gc, impl, ["broadcast_tensor_dict"], strict=True)

    # --- isend_tensor_dict fingerprint + signature ---
    p.check_fingerprint(
        "GroupCoordinator.isend_tensor_dict",
        inspect.getsource(gc.isend_tensor_dict),
        ISEND_TENSOR_DICT,
        resync_hint=(
            "Re-sync _ilu_isend_tensor_dict_sync in "
            "vllm_iluvatar/distributed/parallel_state.py and the "
            "ISEND_TENSOR_DICT baseline in "
            "ilu_patch/general_patches/patch_parallel_state.py to the new "
            "community source of GroupCoordinator.isend_tensor_dict."
        ),
    )

    # --- irecv_tensor_dict fingerprint + signature ---
    p.check_fingerprint(
        "GroupCoordinator.irecv_tensor_dict",
        inspect.getsource(gc.irecv_tensor_dict),
        IRECV_TENSOR_DICT,
        resync_hint=(
            "Re-sync _ilu_irecv_tensor_dict_sync in "
            "vllm_iluvatar/distributed/parallel_state.py and the "
            "IRECV_TENSOR_DICT baseline in "
            "ilu_patch/general_patches/patch_parallel_state.py to the new "
            "community source of GroupCoordinator.irecv_tensor_dict."
        ),
    )

    # Bind community globals referenced by the copied method bodies onto the
    # impl module.
    impl.torch = _torch
    impl.TensorMetadata = _m.TensorMetadata
    impl._split_tensor_dict = _m._split_tensor_dict
    impl.Handle = _m.Handle
    impl.Callable = _abc.Callable

    # Install overrides.
    gc.broadcast = impl._make_broadcast(gc.broadcast)
    gc.broadcast_tensor_dict = impl.broadcast_tensor_dict
    gc.isend_tensor_dict = impl._make_isend_tensor_dict(gc.isend_tensor_dict)
    gc.irecv_tensor_dict = impl._make_irecv_tensor_dict(gc.irecv_tensor_dict)

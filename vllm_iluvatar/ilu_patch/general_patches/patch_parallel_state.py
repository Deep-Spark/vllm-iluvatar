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
           broadcast_tensor_dict}
Baseline : main_4875b4456b5
How      : broadcast            -> wrapper delegating to the community method;
           broadcast_tensor_dict -> full-method copy (in parallel_state.py),
                                    guarded by a source fingerprint against the
                                    baseline snapshot inlined below.
Fingerprint: broadcast_tensor_dict community source is hashed at register-time
           against the BROADCAST_TENSOR_DICT baseline below; gated by
           VLLM_ILU_PATCH_CHECK_FINGERPRINT (default on) so startup fails fast
           if upstream changed the body.
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
# Baseline: main_4875b4456b5
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


def register() -> None:
    """Register ILU GroupCoordinator overrides."""
    p.register_once(
        __name__, _apply, detail="broadcast wrapper, broadcast_tensor_dict copy"
    )


def _apply() -> None:
    import torch as _torch
    import vllm.distributed.parallel_state as _m

    import vllm_iluvatar.distributed.parallel_state as impl

    gc = _m.GroupCoordinator

    # Drift guard for the full-method copy (VLLM_ILU_PATCH_CHECK_FINGERPRINT).
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

    # Strict signature validation for the copied method
    # (VLLM_ILU_PATCH_CHECK_SIGNATURE).
    p.check_signatures(gc, impl, ["broadcast_tensor_dict"], strict=True)

    # Bind community globals referenced by the copied method body onto the impl
    # module.
    impl.torch = _torch
    impl.TensorMetadata = _m.TensorMetadata
    impl._split_tensor_dict = _m._split_tensor_dict

    gc.broadcast = impl._make_broadcast(gc.broadcast)
    gc.broadcast_tensor_dict = impl.broadcast_tensor_dict

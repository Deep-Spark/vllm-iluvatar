# SPDX-License-Identifier: Apache-2.0
"""Completion-only offloading for a worker without local cache payloads."""

from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)


class EmptyOffloadingWorker(OffloadingWorker):
    """Acknowledge global jobs without allocating storage or GPU resources.

    The caller must certify that the rank's canonical cache tensors and data
    references are empty. Other ranks still transfer their own pages; this
    rank contributes a completion to the same collective job accounting.
    """

    def __init__(self) -> None:
        self._finished: dict[int, TransferResult] = {}

    def _submit(self, job_id: int) -> bool:
        if job_id in self._finished:
            raise ValueError(f"Offloading job {job_id} is already pending completion")
        self._finished[job_id] = TransferResult(job_id=job_id, success=True)
        return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._submit(job_id)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return self._submit(job_id)

    def get_finished(self) -> list[TransferResult]:
        finished = list(self._finished.values())
        self._finished.clear()
        return finished

    def wait(self, job_ids: set[int]) -> None:
        # Transfers complete on submission. Leave their completion records for
        # get_finished(), as the ordinary asynchronous worker does after wait.
        return

    def shutdown(self) -> None:
        self._finished.clear()

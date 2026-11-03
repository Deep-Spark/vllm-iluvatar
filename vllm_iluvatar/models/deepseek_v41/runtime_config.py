# SPDX-License-Identifier: Apache-2.0
"""Worker-local runtime policies selected before V4.1 warmup and graph capture."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def configure_w8a8_input_rows(vllm_config: VllmConfig) -> bool:
    """Select GEMM backing storage from the worker's PD role and sequence limit.

    The target is constructed before its draft and before graph capture, so
    both use the same worker-local ixformer setting. Assign both outcomes to
    prevent a producer or another sequence limit from inheriting an enabled
    environment value or a previous model's policy.
    """
    from ixformer.core import config as ixformer_config

    transfer = vllm_config.kv_transfer_config
    enabled = (
        transfer is not None
        and transfer.kv_role == "kv_consumer"
        and vllm_config.scheduler_config.max_num_seqs == 4
    )
    ixformer_config.IXFORMER_W8A8_GEMM_PAD_INPUT_ROWS = enabled
    return enabled

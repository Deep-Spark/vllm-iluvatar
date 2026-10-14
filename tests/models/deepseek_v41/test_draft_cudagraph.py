"""Target and draft CUDA graphs must be independently selectable."""

from types import SimpleNamespace

import pytest
from vllm.config.compilation import CUDAGraphMode

from vllm_iluvatar.v1.worker.gpu.spec_decode.draft_cudagraph import (
    make_init_cudagraph_manager,
)


@pytest.mark.parametrize("eager", [True, False, None])
@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY])
def test_draft_eager_preserves_target_mode(eager, mode):
    runner = SimpleNamespace(vllm_config=SimpleNamespace(
        speculative_config=SimpleNamespace(enforce_eager=eager),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
    ))
    received = []

    def original(self, draft_mode):
        received.append(draft_mode)

    make_init_cudagraph_manager(original)(runner, mode)
    assert received == [CUDAGraphMode.NONE if eager else mode]
    assert runner.vllm_config.compilation_config.cudagraph_mode == mode

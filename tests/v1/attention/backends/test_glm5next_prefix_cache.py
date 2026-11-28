# SPDX-License-Identifier: Apache-2.0
"""CQ GLM prefix matching on mixed MLA, indexer, tail, and KDA cache groups."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams


@pytest.mark.parametrize(
    "cq,arch,enabled,mode,unit,explicit_block,expected,checkpoint",
    [
        (True, "Glm5NextForCausalLM", True, "align", None, None, 256, True),
        (
            True,
            "Glm5NextForConditionalGeneration",
            True,
            "align",
            None,
            4608,
            256,
            True,
        ),
        (True, "Glm5NextForCausalLM", True, "align", 512, None, 512, False),
        (True, "Glm5NextForCausalLM", True, "align", None, 2432, None, False),
        (True, "Glm5NextForCausalLM", False, "align", None, None, None, False),
        (True, "Glm5NextForCausalLM", True, "all", None, None, None, False),
        (False, "Glm5NextForCausalLM", True, "align", None, None, None, False),
        (True, "OtherForCausalLM", True, "align", None, None, None, False),
    ],
)
def test_glm5next_prefix_policy(
    monkeypatch, cq, arch, enabled, mode, unit, explicit_block, expected, checkpoint
):
    from vllm_iluvatar.ilu_util import arch as ilu_arch
    from vllm_iluvatar.platforms.ilu_platform import IluPlatform

    monkeypatch.setattr(ilu_arch, "is_ivcore30", lambda: cq)
    monkeypatch.setattr(ilu_arch, "is_ivcore11", lambda: not cq)
    monkeypatch.setattr(
        IluPlatform.__mro__[1],
        "check_and_update_config",
        classmethod(lambda cls, cfg: None),
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=[arch],
            quantization=None,
            dtype=torch.bfloat16,
            is_hybrid=True,
            hf_text_config=SimpleNamespace(index_kpool=4),
            tokenizer_mode="auto",
            enforce_eager=True,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="bfloat16",
            enable_prefix_caching=enabled,
            mamba_cache_mode=mode,
            prefix_match_unit=unit,
            enable_mamba_shared_prefix_checkpoint=False,
            user_specified_block_size=explicit_block is not None,
            block_size=explicit_block or 64,
        ),
        quant_config=None,
        additional_config=None,
        load_config=SimpleNamespace(load_format="auto"),
        compilation_config=None,
        scheduler_config=None,
        parallel_config=SimpleNamespace(worker_cls="custom"),
    )
    IluPlatform.check_and_update_config(config)
    assert config.cache_config.prefix_match_unit == expected
    assert config.cache_config.enable_mamba_shared_prefix_checkpoint is checkpoint
    if checkpoint:
        assert IluPlatform._get_indexer_block_alignment(config) % expected == 0


@pytest.mark.parametrize(
    "state_blocks", [0, 3], ids=["record-fold", "per-draft-states"]
)
@pytest.mark.parametrize(
    "block_size,kv_dtype,cache_dtype,state_dtype",
    [
        (1280, torch.bfloat16, "auto", torch.bfloat16),
        (4608, torch.float8_e4m3fn, "fp8_e4m3", torch.float32),
    ],
)
def test_glm5next_shared_prefix_hits(
    block_size, kv_dtype, cache_dtype, state_dtype, state_blocks
):
    """Exercise real vLLM MTP allocation, hit lookup, and slot reuse on a warm prefix."""
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        _get_kv_cache_groups_glm5_next,
        generate_scheduler_kv_cache_config,
        get_kv_cache_config_from_groups,
        get_request_block_hasher,
        init_none_hash,
        resolve_kv_cache_block_sizes,
    )
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        KpoolTailSpec,
        MambaSpec,
        MLAAttentionSpec,
        get_kv_quant_mode,
    )
    from vllm.v1.request import Request

    match_unit = 256
    num_spec = 3
    max_batched = 8192

    mla_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        max_tp_shards=1,
        head_size=512,
        dtype=kv_dtype,
        cache_dtype_str=cache_dtype,
        kv_quant_mode=get_kv_quant_mode(cache_dtype),
        is_index_group_leader=True,
        block_stride_alignment=512,
    )
    idx_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
        storage_block_size=256,
    )
    tail_spec = KpoolTailSpec(
        block_size=8,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=8,
    )
    mamba_spec = MambaSpec(
        shapes=((12288, 6), (32, 128, 128)),
        dtypes=(torch.bfloat16, state_dtype),
        block_size=block_size,
        mamba_cache_mode="align",
        num_speculative_blocks=state_blocks,
    )
    specs = {
        "mla.0": mla_spec,
        "idx.0": idx_spec,
        "tail.0": tail_spec,
        "mamba.0": mamba_spec,
        "mamba.1": mamba_spec,
        "mamba.2": mamba_spec,
    }
    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=276480,
            get_total_num_hidden_layers=lambda: 6,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            prefix_match_unit=match_unit,
            prefix_cache_retention_interval=None,
            num_gpu_blocks_override=512,
            block_size=block_size,
        ),
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=None,
    )
    groups = _get_kv_cache_groups_glm5_next(vllm_cfg, specs)
    assert groups is not None
    kv_cfg = get_kv_cache_config_from_groups(vllm_cfg, groups, available_memory=1 << 34)
    sched_kv_cfg = generate_scheduler_kv_cache_config([kv_cfg])
    sched_bs, hash_bs = resolve_kv_cache_block_sizes(sched_kv_cfg, vllm_cfg)
    assert (sched_bs, hash_bs) == (block_size, match_unit)

    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    hasher = get_request_block_hasher(hash_bs, hash_fn)

    def _run_prefill(mgr: KVCacheManager, req: Request) -> None:
        stub = SimpleNamespace(
            block_size=sched_bs,
            cache_config=SimpleNamespace(block_size=sched_bs),
            use_eagle_block_drop=True,
            max_num_scheduled_tokens=max_batched,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
            mamba_partial_cache_hit=True,
            mamba_shared_prefix_checkpoint=mgr.mamba_shared_prefix_checkpoint,
            hash_block_size=hash_bs,
            mamba_has_prefill_checkpoint_blocks=False,
            mamba_prefill_checkpoint_alignment=None,
        )
        while req.num_computed_tokens < req.num_tokens:
            rem = req.num_tokens - req.num_computed_tokens
            step_toks = Scheduler._mamba_block_aligned_split(
                stub, req, min(rem, max_batched)
            )
            assert step_toks > 0
            assert (
                mgr.allocate_slots(req, step_toks, num_lookahead_tokens=num_spec)
                is not None
            )
            req.num_computed_tokens += step_toks
            mgr.cache_blocks(req, req.num_computed_tokens)
            mgr.take_kv_cache_block_copies()
            mgr.new_step_starts()

    for prompt_len, expected_hit in (
        (8192 + 12, 8192 - 256),
        (65536 + 12, 65536 - 256),
        (262144 + 12, 262144 - 256),
    ):
        mgr = KVCacheManager(
            kv_cache_config=sched_kv_cfg,
            max_model_len=276480,
            scheduler_block_size=sched_bs,
            hash_block_size=hash_bs,
            enable_caching=True,
            use_eagle=True,
            enable_mamba_shared_prefix_checkpoint=True,
        )
        tokens = [(i * 7 + 3) % 30000 for i in range(prompt_len)]
        req_warm = Request(
            request_id="warm",
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=16),
            pooling_params=None,
            block_hasher=hasher,
        )
        _run_prefill(mgr, req_warm)

        req_hit = Request(
            request_id="hit",
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=16),
            pooling_params=None,
            block_hasher=hasher,
        )
        computed_blocks, num_hit, _ = mgr.get_computed_blocks(req_hit)
        assert num_hit == expected_hit
        rem = req_hit.num_tokens - num_hit
        assert (
            mgr.allocate_slots(
                req_hit,
                rem,
                num_new_computed_tokens=num_hit,
                new_computed_blocks=computed_blocks,
                num_lookahead_tokens=num_spec,
            )
            is not None
        )

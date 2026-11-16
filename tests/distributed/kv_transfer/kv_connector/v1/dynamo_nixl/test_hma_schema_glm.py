# SPDX-License-Identifier: Apache-2.0
"""Map GLM-5.2 MLA + DSA indexer layers onto the existing HMA role vocabulary.

KVBM's payload whitelist is DSV4-shaped (CSA/K_Comp, CSA/K_IComp). GLM-5.2
reuses those roles as Prefix-only groups; no junction / sliding_window.
The empty-ratio GLM fallback is gated on ``model_type == "glm_moe_dsa"``.
DeepSeek-V3.2 uses the same ``self_attn.attn`` / ``self_attn.indexer.k_cache``
names and must stay on the non-HMA path.
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm_iluvatar.distributed.kv_transfer.kv_connector.v1.dynamo_nixl.schema import (
    build_kv_cache_schema,
    export_engine_group_metadata,
    is_hma_model_config,
)


class _Spec:
    def __init__(self, block_size: int, page_size_bytes: int, **extra):
        self.block_size = block_size
        self.page_size_bytes = page_size_bytes
        for name, value in extra.items():
            setattr(self, name, value)


def _group(layer_names: tuple[str, ...], spec: _Spec) -> SimpleNamespace:
    return SimpleNamespace(kv_cache_spec=spec, layer_names=layer_names)


def _schema(groups: tuple[SimpleNamespace, ...], *, model_type: str | None = None):
    return build_kv_cache_schema(
        SimpleNamespace(kv_cache_groups=groups, kv_cache_tensors=()),
        model_type=model_type,
    )


def _roles(schema):
    return {
        (entry.cache_family, entry.cache_role, entry.sliding_window)
        for group in schema.groups
        for entry in group.entries
    }


def test_glm_moe_dsa_maps_mla_and_indexer_to_prefix_csa():
    schema = _schema(
        (
            _group(
                ("model.layers.0.self_attn.attn",),
                _Spec(64, 73728),
            ),
            _group(
                ("model.layers.0.self_attn.indexer.k_cache",),
                _Spec(64, 16896),
            ),
        ),
        model_type="glm_moe_dsa",
    )

    assert _roles(schema) == {("CSA", "K_Comp", None), ("CSA", "K_IComp", None)}
    assert all(group.external and group.reusable for group in schema.groups)


def test_glm_moe_dsa_accepts_uppercase_model_type():
    schema = _schema(
        (
            _group(("model.layers.0.self_attn.attn",), _Spec(64, 73728)),
            _group(("model.layers.0.self_attn.indexer.k_cache",), _Spec(64, 16896)),
        ),
        model_type="GLM_MOE_DSA",
    )
    assert _roles(schema) == {("CSA", "K_Comp", None), ("CSA", "K_IComp", None)}


def test_deepseek_v32_self_attn_indexer_is_not_csa():
    """DeepSeek-V3.2 reuses GLM's cache names; model_type must keep it off CSA."""
    groups = (
        _group(("model.layers.0.self_attn.attn",), _Spec(64, 73728)),
        _group(("model.layers.0.self_attn.indexer.k_cache",), _Spec(64, 16896)),
    )
    schema = _schema(groups, model_type="deepseek_v32")
    assert _roles(schema) == {("unknown", "unknown", None)}
    assert all(not group.external and not group.reusable for group in schema.groups)


def test_non_positive_tokens_per_state_is_not_compression():
    schema = _schema(
        (
            _group(
                ("model.layers.0.attn.swa_cache",),
                _Spec(64, 8192, tokens_per_state=-1),
            ),
        ),
        model_type="deepseek_v4",
    )
    entry = schema.groups[0].entries[0]
    assert entry.compress_ratio == 1
    assert entry.storage_block_size == 64
    assert entry.cache_family == "SWA"


def test_dsv4_tokens_per_state_derives_storage_block_size():
    schema = _schema(
        (
            _group(
                ("model.layers.3.attn.ilu_deepseek_v4_multi_head_latent_attention",),
                _Spec(256, 8192, tokens_per_state=4),
            ),
        ),
        model_type="deepseek_v4",
    )
    entry = schema.groups[0].entries[0]
    assert entry.compress_ratio == 4
    assert entry.storage_block_size == 64
    assert entry.cache_family == "CSA"
    assert entry.cache_role == "K_Comp"


def test_dsv4_tokens_per_state_roles():
    schema = _schema(
        (
            _group(
                ("model.layers.3.attn.ilu_deepseek_v4_multi_head_latent_attention",),
                _Spec(256, 8192, tokens_per_state=4),
            ),
            _group(
                ("model.layers.3.attn.indexer.k_cache",),
                _Spec(256, 8192, tokens_per_state=4),
            ),
        ),
        model_type="deepseek_v4",
    )

    assert _roles(schema) == {("CSA", "K_Comp", None), ("CSA", "K_IComp", None)}


def test_engine_metadata_export_carries_schema_hash():
    schema = _schema(
        (
            _group(
                ("model.layers.3.attn.ilu_deepseek_v4_multi_head_latent_attention",),
                _Spec(256, 8192, tokens_per_state=4),
            ),
        ),
        model_type="deepseek_v4",
    )
    rows = export_engine_group_metadata(schema)
    assert rows
    assert rows[0]["version"] == 1
    assert rows[0]["schema_hash"]
    assert rows[0]["kind"]
    assert rows[0]["entries"]


def test_hma_model_config_reads_hf_config():
    assert is_hma_model_config(
        SimpleNamespace(
            model_config=SimpleNamespace(
                model_type=None,
                hf_config=SimpleNamespace(model_type="deepseek_v4"),
            )
        )
    )
    assert not is_hma_model_config(
        SimpleNamespace(
            model_config=SimpleNamespace(
                model_type="qwen3",
                hf_config=SimpleNamespace(model_type="qwen3"),
            )
        )
    )

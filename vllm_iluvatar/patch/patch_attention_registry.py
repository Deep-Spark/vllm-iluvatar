"""Add iluvatar attention backends to ``AttentionBackendEnum``.

The internal fork adds three named members to
``vllm.v1.attention.backends.registry.AttentionBackendEnum``::

    ILU_FLASH_ATTN     -> vllm.v1.attention.backends.ilu_flash_attn.IluFlashAttentionBackend
    ILU_MLA            -> vllm.v1.attention.backends.mla.ilu_mla.IluMLABackend
    ILU_FLASHMLA_SPARSE-> vllm.v1.attention.backends.mla.ilu_flashmla_sparse.IluFlashMLASparseBackend

Community vLLM only exposes a single ``CUSTOM`` placeholder, which is not
enough for three distinct backends selected by name from the (patched)
platform routing. We therefore inject real enum members at runtime.

The backend *class paths* point at stock ``vllm.*`` module paths which the
override finder (:mod:`vllm_iluvatar.patch.patch_sys_modules`) resolves to our
mirror-tree copies, so no extra wiring is needed for ``get_class()``.
"""

from __future__ import annotations

_PATCHED = False

_ILU_BACKENDS = {
    "ILU_FLASH_ATTN": (
        "vllm.v1.attention.backends.ilu_flash_attn.IluFlashAttentionBackend"
    ),
    "ILU_MLA": "vllm.v1.attention.backends.mla.ilu_mla.IluMLABackend",
    "ILU_FLASHMLA_SPARSE": (
        "vllm.v1.attention.backends.mla.ilu_flashmla_sparse." "IluFlashMLASparseBackend"
    ),
}


def _add_enum_member(enum_cls, name: str, value: str):
    """Add a new member to an already-created Enum class (CPython).

    Enums are normally immutable after creation; this mutates the internal
    member maps the same way ``EnumMeta`` does during class body execution.
    """
    if name in enum_cls.__members__:
        return enum_cls.__members__[name]

    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    # Register in the lookup maps and as a class attribute.
    enum_cls._member_map_[name] = member
    # _value2member_map_ may reject unhashable values; backend paths are str.
    try:
        enum_cls._value2member_map_[value] = member
    except TypeError:
        pass
    enum_cls._member_names_.append(name)
    type.__setattr__(enum_cls, name, member)
    return member


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from vllm.logger import init_logger
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    logger = init_logger(__name__)

    added = []
    for name, path in _ILU_BACKENDS.items():
        _add_enum_member(AttentionBackendEnum, name, path)
        added.append(name)

    logger.info("vllm-iluvatar: registered attention backends %s", ", ".join(added))

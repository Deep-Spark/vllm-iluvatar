"""All monkey-patches and ``sys.modules`` redirects required to bridge
stock vLLM (iluvatar_main branch) to the ilu DeepSeek-V4 W4A8 path.

Each ``patch_*`` module exports a single ``apply()`` callable; the
plugin's :func:`vllm_iluvatar.utils.adapt_patch` orchestrates the call
order. Module redirects live in :mod:`patch_sys_modules` and must run
before any attribute-level patch (otherwise an attribute might be set on
the soon-to-be-replaced module rather than the replacement).
"""

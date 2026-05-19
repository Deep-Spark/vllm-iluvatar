# vllm-iluvatar

> Out-of-tree vLLM plugin for Iluvatar GPUs

[English](README.md) | [简体中文](README.zh.md)

## Project Overview

`vllm-iluvatar` is a hardware-adapter plugin built on vLLM's official
[plugin system](https://docs.vllm.ai/en/latest/design/plugin_system.html).
It hooks into the `vllm.general_plugins` entry point so vLLM picks it up
automatically at startup -- no edits to the vLLM main repo required -- and
enables **DeepSeek-V4-Flash W4A8 (v1-TN)** inference on Iluvatar GPUs.

**Current scope.** The plugin only carries the incremental changes required
for the `DeepSeek-V4-Flash w4a8 (v1-TN)` variant. Every other model /
feature already supported on Iluvatar is still maintained in the
`iluvatar_main` fork of vLLM.

## Changelog

- **[2026-05] v0.1.0** — first release. Adds support for the
  `IluDeepseekV4ForCausalLM` model architecture and the
  `ilu_deepseek_v4_w4a8` quantization method via 6 `sys.modules`
  redirects + 4 attribute-level monkey patches that bridge upstream vLLM
  to the DeepSeek-V4 W4A8 path.

## Prerequisites

- **Hardware:** Iluvatar GPU
- **OS:** Linux
- **Software:**
  - Python `>= 3.10`
  - vLLM: must be on the **`iluvatar_main` branch**
  - PyTorch / ixformer / ixinfer: keep in sync with the version pinned
    by the vLLM `iluvatar_main` branch

## Supported Models

| Model | Quantization | Architecture / Notes |
| --- | --- | --- |
| DeepSeek-V4-Flash | W4A8 (v1-TN) | `IluDeepseekV4ForCausalLM` + `ilu_deepseek_v4_w4a8` |

## Installation

In the snippets below, `{{VLLM_REPO}}` is your local vLLM checkout and
`{{PLUGIN_REPO}}` is this repository.

```bash
# 1. Make sure vLLM is on iluvatar_main (no DSv4 patch applied)
cd {{VLLM_REPO}}
git checkout iluvatar_main

# 2. Editable install of the plugin
cd {{PLUGIN_REPO}}
pip install -e .

# 3. Self-check (no checkpoint needed, runs on CPU)
python scripts/smoke_test.py
```

`scripts/smoke_test.py` runs 6 assertions: the entry point is
discoverable, the model architecture is registered, the quantization
method is registered, the 6 `sys.modules` redirects are in place, the
`vllm._custom_ops` patch is applied, and `ModelConfig.__post_init__` is
wrapped. The install is healthy only when **every line prints `[OK  ]`**.

## Development

Set up the pre-commit hooks once after cloning so every commit auto-runs
black / isort and the standard text-hygiene checks (trailing whitespace,
LF line endings, large-file guard, ...):

```bash
pip install pre-commit
pre-commit install
```

Run the hooks against the entire repo on demand:

```bash
pre-commit run --all-files
```

## Quick Start

> Defaults match the `config.sh` shipped under
> `inference/vllm/models/DeepSeek-V4/DeepSeek-V4-Flash-w4a8-v1-TN/`
> in the `llm-modelzoo` repository.

### Online inference

```bash
vllm serve {{MODEL_PATH}}/DeepSeek-V4-Flash-w4a8-v1-TN \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 2048 \
  --port 12345
```

Health probe:

```bash
curl http://127.0.0.1:12345/v1/models
```

Sample request:

```bash
curl http://127.0.0.1:12345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "{{MODEL_PATH}}/DeepSeek-V4-Flash-w4a8-v1-TN",
    "messages": [{"role": "user", "content": "Briefly introduce DeepSeek."}],
    "temperature": 0.0,
    "max_tokens": 128
  }'
```

### One-shot scripts

The `llm-modelzoo` repo ships wrappers that handle "start server →
probe → run eval/bench → tear down":

```bash
cd {{LLM_MODELZOO_REPO}}/inference/vllm/models/DeepSeek-V4/DeepSeek-V4-Flash-w4a8-v1-TN

bash run_mmlu_accuracy.sh           # MMLU accuracy
bash run_perf_serving.sh            # perf bench (1-stream 2048→1024 by default)
START_SERVER=0 bash run_perf_serving.sh --max-concurrency 64 --num-prompts 256
```

## Verify the Plugin is Loaded

### Method 1: grep the server log

A successful `vllm serve` startup logs the following 5 `vllm-iluvatar:`
lines:

```text
vllm-iluvatar: installed sys.modules redirects for 6 modules
vllm-iluvatar: patched vllm._custom_ops (silu_and_mul_with_clamp, router_gemm_bf16_fp32, topk_hash_softplus_sqrt, dsa_convert_req_index_to_global_index)
vllm-iluvatar: patched vllm.v1.core.kv_cache_utils (_get_kv_cache_config_deepseek_v4 + get_kv_cache_config_from_groups)
vllm-iluvatar: defaulting tokenizer_mode='deepseek_v4' for IluDeepseekV4ForCausalLM
vllm-iluvatar: registered model architecture IluDeepseekV4ForCausalLM
```

If none of them show up the plugin was not installed or not loaded —
see the FAQ below.

### Method 2: runtime probe

```python
import vllm.plugins as p
p.load_general_plugins()

from vllm import ModelRegistry
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

assert "IluDeepseekV4ForCausalLM" in ModelRegistry.get_supported_archs()
assert "ilu_deepseek_v4_w4a8" in QUANTIZATION_METHODS
```

## Further Reading

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — plugin load order, the
  `patch/` vs. `_replacements/` split, the inventory of the 6 full-file
  replacements with rationales, and an upgrade guide
- [vLLM plugin system docs](https://docs.vllm.ai/en/latest/design/plugin_system.html)
- [vLLM model registration](https://docs.vllm.ai/en/stable/contributing/model/registration/)

## License

Apache-2.0 — see [LICENSE](LICENSE).

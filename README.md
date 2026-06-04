<h3 align="center">
vLLM Iluvatar Plugin
</h3>

<p align="center">
| <a href="docs/ARCHITECTURE.md"><b>Architecture</b></a> | <a href="https://docs.vllm.ai/en/latest/design/plugin_system.html"><b>vLLM Plugin System</b></a> |
</p>

<p align="center">
<b>English</b> | <a href="README.zh.md">中文</a>
</p>

---

## Latest News

- [2026-06] v0.2.0 — 50+ models supported.
- [2026-05] v0.1.0 — first release, DeepSeek-V4 only.

---

## Overview

`vllm-iluvatar` is an out-of-tree plugin that lets vLLM run on Iluvatar GPUs.

## Prerequisites

- **Hardware:** Iluvatar GPU
- **OS:** Linux
- **Software:**
  - Python `>= 3.10`
  - vLLM: community release
  - PyTorch / ixformer / ixinfer: in sync with the Iluvatar runtime

## Getting Started

Below, `{{PLUGIN_REPO}}` is this repository.

```bash
# 1. Install community vLLM

# 2. Editable install of the plugin (on top of community vLLM)
cd {{PLUGIN_REPO}}
pip install -e .

# 3. (Recommended on Iluvatar) install the interpreter-startup bootstrap so the
#    mirror-tree finder is active before `import vllm` -- required to keep the
#    ixformer communication optimization.
python3 scripts/install_bootstrap.py

```

## Run a Model

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

Health probe and a sample request:

```bash
curl http://127.0.0.1:12345/v1/models

curl http://127.0.0.1:12345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "{{MODEL_PATH}}/DeepSeek-V4-Flash-w4a8-v1-TN",
    "messages": [{"role": "user", "content": "Briefly introduce DeepSeek."}],
    "temperature": 0.0,
    "max_tokens": 128
  }'
```

## License

Apache-2.0 — see [LICENSE](LICENSE).

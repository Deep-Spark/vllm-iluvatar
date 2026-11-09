# vLLM Iluvatar Plugin

![vLLM × Iluvatar CoreX](docs/assets/logos/vllm-iluvatar.png)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/vLLM-0.27.1-green)](https://github.com/vllm-project/vllm)

vLLM Iluvatar is a hardware plugin that enables [vLLM](https://github.com/vllm-project/vllm)
inference and serving on Iluvatar CoreX GPUs. The plugin is automatically
loaded by vLLM after installation.

## Requirements

- Linux with an Iluvatar CoreX GPU and a matching driver and SDK.
- Python 3.10 or newer, CMake, and Ninja.
- A compatible CoreX software environment:

| Component | Version |
| --- | --- |
| vLLM | `>=0.27.1,<0.28.0` |
| CoreX runtime and driver | `>=5.0.0` |
| CoreX PyTorch | `>=2.10.0` |
| ixformer | `>=0.7.0` |

Install the matching CoreX PyTorch, Triton, ixformer, and vLLM packages before
building this plugin. CoreX-specific packages are supplied separately and are
not included as Git submodules or installed from public PyPI by this project.

## Installation

From the repository root, select your GPU architecture and build the wheel:

```bash
# Use the architecture of your GPU, for example ivcore11 or ivcore30.
export COREX_ARCH=ivcore11
bash build_vllm_iluvatar.sh
VLLM_ILUVATAR_INSTALL_RUNTIME_DEPS=1 bash install_vllm_iluvatar.sh
```

Verify the native extension and platform registration:

```bash
python3 -c "import torch; import vllm_iluvatar._C; from vllm.platforms import current_platform; print(current_platform)"
```

The output should identify the Iluvatar platform. For editable installation,
see the [development guide](docs/developer_guide/guides/7-how-to-set-up-the-dev-environment.md).

## Quick Start

Start an OpenAI-compatible server with a model supported by your CoreX environment:

```bash
vllm serve /path/to/model --port 8000
```

Send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/model","messages":[{"role":"user","content":"Hello!"}]}'
```

## Kernel provider

If a different kernel provider is needed,
set `VLLM_ILU_KERNEL_PROVIDER` before starting vLLM. The process uses one
backend throughout loading, repacking, warmup, capture, inference, and replay.

| Value | Behavior |
| --- | --- |
| `baseline` | Default. Use the original ixformer, Torch, Triton/FLA, and plugin CUDA implementations; startup does not require ixaite. |
| `ixaite` | Use the implementations in `vllm_iluvatar/backends/ixaite`. Unsupported operators and formats fail explicitly before modifying weights, cache, or state. |

Ixaite-specific implementation code must live only in
`vllm_iluvatar/backends/ixaite/`. Related tests and benchmarks must live under
`tests/ixaite/`, including `tests/ixaite/benchmarks/`.

```bash
VLLM_ILU_KERNEL_PROVIDER=ixaite vllm serve /path/to/model
```

## Documentation

- [Development setup](docs/developer_guide/guides/7-how-to-set-up-the-dev-environment.md)
- [Model integration](docs/developer_guide/guides/2-how-to-quickly-adapt-a-community-model.md)
- [Quantization](docs/developer_guide/guides/8-how-to-use-iluvatar-runtime-quant-via-additional-config.md)
- [Distributed inference](docs/developer_guide/guides/5-how-to-use-dp-ep.md)
- [Plugin design](docs/developer_guide/design/iluvatar_plugin_design.md) — Detailed implementation notes
- [Changelog](changelog.md)

For general serving options and API usage, see the [vLLM documentation](https://docs.vllm.ai/).

## License

This project is licensed under the [Apache License 2.0](LICENSE).

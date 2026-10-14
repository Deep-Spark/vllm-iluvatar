# vLLM Iluvatar Plugin

![vLLM × Iluvatar CoreX](docs/assets/logos/vllm-iluvatar.png)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![vLLM](https://img.shields.io/badge/vLLM-0.31.0-green)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

vLLM Iluvatar is a hardware plugin that enables [vLLM](https://github.com/vllm-project/vllm)
inference and serving on Iluvatar CoreX GPUs. It adds platform integration,
`ixformer`-backed communication and IR operators, selected `torch.ops._C`
compatibility bindings, model and quantization support, runtime patches, and
NIXL KV-transfer connectors — without modifying the upstream vLLM source tree.
The plugin is loaded automatically by vLLM after installation.

## Requirements

- Linux with an Iluvatar CoreX GPU and a matching driver and SDK.
- Python 3.10 or newer, CMake, and Ninja.
- A compatible CoreX software environment:

| Component | Version |
| --- | --- |
| vLLM | `>=0.31.0,<0.32.0` |
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

`_C.so` contains CUDA kernels, so it must be compiled for the ivcore
architecture of the machine it will run on. When `COREX_ARCH` is not set,
`setup.py` falls back to `ivcore11`; a wrong-arch extension loads without error
but returns wrong values (degenerate text, no crash), so set it explicitly.

Verify the native extension and platform registration:

```bash
python3 -c "import torch; import vllm_iluvatar._C; from vllm.platforms import current_platform; print(current_platform)"
```

The output should identify the Iluvatar platform. For editable installation,
see the [development guide](docs/guides/7-how-to-set-up-the-dev-environment.md).

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

For NIXL KV transfer, use vLLM's `--kv-transfer-config` option to select
`IluNixlConnector`, `IluNixlPullConnector`, or `IluNixlPushConnector`,
depending on the deployment topology.

## Build a runtime image (Docker)

The repository ships a multi-stage `Dockerfile` that builds the whole stack from
source: it clones and compiles ixinfer, ixformer, and vLLM at the commits pinned
in the Dockerfile build args, builds this plugin from the checked-out source, and
produces a runnable CoreX runtime image (SDK + torch / triton + source-built
`libcuinfer` + the three wheels).

```bash
# Needs SSH access to the source repositories for the pinned clones.
bash scripts/build_image.sh
```

The image is tagged for a container registry with a CPU-arch and build-date
suffix (`<base-tag>-<arch>-<YYYYMMDD>`, arch defaulting to `uname -m`); override
the target via the `REGISTRY` / `PROJECT` / `BASE_TAG` / `ARCH` / `BUILD_DATE`
environment variables. The CoreX driver is deliberately **not** baked into the
image; install it inside the (privileged) container before serving — see the
comments at the top of the `Dockerfile`.

## Documentation

- [Plugin design](docs/design/iluvatar_plugin_design.md)
- [Development setup](docs/guides/7-how-to-set-up-the-dev-environment.md)
- [Model integration](docs/guides/2-how-to-quickly-adapt-a-community-model.md)
- [Quantization](docs/guides/8-how-to-use-iluvatar-runtime-quant-via-additional-config.md)
- [Distributed inference](docs/guides/5-how-to-use-dp-ep.md)
- [Changelog](changelog.md)

For general serving options and API usage, see the [vLLM documentation](https://docs.vllm.ai/).

## License

This project is licensed under the [Apache License 2.0](LICENSE).

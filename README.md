# vLLM Iluvatar Plugin

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![vLLM Plugin](https://img.shields.io/badge/vLLM-plugin-green)](https://github.com/vllm-project/vllm)

`vllm_iluvatar` is an out-of-tree [vLLM](https://github.com/vllm-project/vllm)
plugin for the Iluvatar CoreX platform. It adds platform integration,
`ixformer`-backed communication and IR operators, selected `torch.ops._C`
compatibility bindings, model and quantization support, runtime patches, and
NIXL KV-transfer connectors—without modifying the upstream vLLM source tree.

## Architecture Overview

vLLM loads the plugin through two entry points:

- `vllm.platform_plugins`: `register_platform()` activates `IluPlatform`,
  early compatibility shims, runtime configuration registration, attention backend
  registration, and worker startup hooks.
- `vllm.general_plugins`: `register_general()` applies late runtime patches and
  registers out-of-tree `CustomOp` implementations, models, quantization
  methods, and KV-transfer connectors.

The package includes:

- `IluPlatform`, a CUDA-compatible CoreX platform adapter.
- `IluCommunicator`, backed by `ixformer.distributed` collectives.
- `vllm_iluvatar._C`, a native extension providing vLLM-compatible `_C`
  operators such as `rms_norm`, `weak_ref_tensor`, and
  `get_cuda_view_from_cpu_tensor`.
- `vllm_iluvatar._onchip_check`, an optional SLB/register/L1/L2 integrity
  checker integrated with the GPU Worker lifecycle.
- `vllm_iluvatar.kernels`, an Iluvatar IR operator provider for vLLM V1.
- Runtime import hooks and compatibility shims required during worker startup.
- Out-of-tree model and quantization support for Qwen3 and DeepSeek-V4.
- `IluNixlConnector`, `IluNixlPullConnector`, and `IluNixlPushConnector` for
  disaggregated prefill/decode deployments.

For implementation details, see the [plugin design](docs/design/iluvatar_plugin_design.md).

## Quick Start

### Prerequisites

- An Iluvatar CoreX GPU environment with `corex`, `torch`, and `ixformer`.
- Python 3.10 or newer.
- A vLLM version compatible with this plugin branch.
- CMake and Ninja for building the native extension.

The CoreX environment must provide `torch` and `ixformer`; this package does
not install them.

### Install vLLM

Install the upstream vLLM version that matches this plugin branch and your
runtime environment:

```bash
pip3 install vllm --no-deps
```

### Install vllm_iluvatar

For development, use an editable installation:

```bash
bash install_vllm_iluvatar_editable.sh
```

For a wheel-based installation:

```bash
bash build_vllm_iluvatar.sh
bash install_vllm_iluvatar.sh
```

#### Target architecture

`_C.so` contains CUDA kernels (RoPE, KV-cache ops), so it must be compiled for
the ivcore architecture of the machine it will run on. `setup.py` resolves the
target in this order (same as ixformer / ixmm):

```text
CUDA_ARCH -> COREX_ARCH -> CMAKE_CUDA_ARCHITECTURES
  -> TORCH_CUDA_ARCH_LIST -> torch.corex_arch -> "ivcore11"
```

The final fallback makes a build silently target `ivcore11` whenever `torch` is
not importable in the build environment. A wrong-arch `_C.so` does not fail to
load: the kernels run and return wrong values, which surfaces much later as a
model that generates degenerate text with no crash and no NaN. Pass the
architecture explicitly:

```bash
COREX_ARCH=ivcore30 bash install_vllm_iluvatar_editable.sh
```

### Verify

```bash
python3 -c "from vllm.platforms import current_platform; print(current_platform)"
```

The output or startup logs should confirm that the Iluvatar platform plugin is
active.

Also confirm the extension was built for this machine. The build directory name
carries the architecture, so compare it against the hardware:

```bash
python3 -c "import torch; print('hardware:', torch.corex_arch)"
ls -d build/Release-*        # e.g. Release-py312-ivcore30
```

A mismatch means `_C.so` must be rebuilt with `COREX_ARCH` set; remove the stale
`build/Release-*` directory first so the old objects are not reused.

## Usage

After installing the plugin, launch models with the standard `vllm serve`
command. Model-specific launch scripts can define arguments in a `SERVE_ARGS`
array and pass them through together with the serving port:

```bash
vllm serve /path/to/model "${SERVE_ARGS[@]}" --port 8000
```

Example launch configurations:

```bash
# Qwen3-14B dense
vllm serve /path/to/qwen3_model \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 2048 \
  --no-async-scheduling

# DeepSeek-V4-Flash W4A8 (TN layout)

vllm serve /path/to/deepseek_v4_model \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 2048
```

For NIXL KV transfer, use vLLM's `--kv-transfer-config` option to select
`IluNixlConnector`, `IluNixlPullConnector`, or `IluNixlPushConnector`, depending
on the deployment topology.

### On-chip storage integrity checking

The checker is disabled by default. Enable it before starting vLLM:

```bash
export VLLM_ILU_ONCHIP_CHECK=1
vllm serve /path/to/model ...
```

When enabled, every GPU Worker runs a strict check after model warmup, one
synchronous check before each non-empty GPU batch, and a log-only check during
shutdown. `FAULT` and CUDA failures stop startup or online execution.
`INCOMPLETE` means coverage was inconclusive rather than successful: startup
retries and then fails, while online execution logs it with rate limiting and
continues the current batch.

Optional settings:

- `VLLM_ILU_ONCHIP_CHECK_LEVEL_MASK` — bit mask for SLB/REG/L1/L2; `0` tests all.
- `VLLM_ILU_ONCHIP_CHECK_KMSG` — report faults through `/dev/itrctl` XID.
- `VLLM_ILU_ONCHIP_CHECK_STARTUP_RETRIES` — default `2`.
- `VLLM_ILU_ONCHIP_CHECK_RETRY_DELAY_MS` — default `100`.
- `VLLM_ILU_ONCHIP_CHECK_INCOMPLETE_LOG_INTERVAL` — default `100`.

The full check retains about 17 MiB per Worker and synchronizes its compute
stream on every checked batch (about 180 microseconds on BI-V150 under an idle
GPU). A BI-V150 exposes its two dies as two GPU devices, so each associated
Worker maintains and runs its own checker.

## Documentation

- [Plugin Design](docs/design/iluvatar_plugin_design.md) — Detailed implementation notes

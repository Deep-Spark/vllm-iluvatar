# 启动参数和环境变量



本页说明 `vllm-iluvatar` 常用启动参数和少量调试环境变量。实际参数应结合模型大小、并行方式、显存容量和测试目标调整。

服务与接口遵循上游 vLLM 的 OpenAI 兼容 server，本版未改动默认端口、endpoint 和请求参数；接口细节见 [vLLM 官方文档](https://docs.vllm.ai/)。本页只列本版相关的常用参数与约束。

## 常用启动参数

### 服务地址



```bash
--host 0.0.0.0
--port 12345
```

`--host 0.0.0.0` 表示监听所有网卡；本机验证时也可以只访问 `127.0.0.1:12345`。

### 模型名称



```bash
--served-model-name test
```

该名称会作为 OpenAI API 请求中的 `model` 字段使用：

```json
{
  "model": "test",
  "messages": []
}
```

### 并行参数



```bash
--tensor-parallel-size 8
--data-parallel-size 1
```

- `--tensor-parallel-size`：张量并行规模。
- `--data-parallel-size`：数据并行规模，普通单服务验证可保持为 `1`。

并行规模需要与模型、设备数量和部署方案匹配。

### 显存使用比例



```bash
--gpu-memory-utilization 0.8
```

控制 vLLM 可使用的显存比例。快速验证可从较保守的值开始，压测或大模型部署时再结合显存情况调整。

## 显存与并发 {#memory-and-concurrency}



以下参数均为上游 vLLM 启动参数，本版没有另行核定通用推荐值；取值与权衡遵循上游 vLLM。

- `--gpu-memory-utilization`：控制 vLLM 可使用的显存比例。
- `--max-num-seqs`：控制一次调度中可并发处理的最大序列数。
- `--max-model-len`：控制模型可接受的最大上下文长度。
- `--max-num-batched-tokens`：控制一次批处理可容纳的 token 数。

DP / EP 场景的参数约束见[数据并行与专家并行](data_parallel_expert_parallel.md)，CUDA Graph 相关约束见 [CUDA Graph](cuda_graph.md)。

### CUDA Graph / eager 模式



`--enforce-eager` 只作调试兜底：

```bash
--enforce-eager
```

CUDA Graph 只支持显式 `mode` 0 + `FULL_DECODE_ONLY`：

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}'
```

### 指向功能页的参数



- `--speculative-config` 见 [DSpark 投机解码](dspark_speculative_decoding.md)。
- `--quantization ilu_runtime` 见 [量化](quantization.md)。
- `--config` 见 [量化](quantization.md)。
- `--enable-expert-parallel` 见 [数据并行与专家并行](data_parallel_expert_parallel.md)。

## 异步调度



本版异步调度默认开启；旧 launch 脚本里的 `--no-async-scheduling` 一律去掉。

仍需保留 `--no-async-scheduling` 的例外有两种：

- 开启天数 INT8 KV cache（设了 `VLLM_ILU_FLASH_ATTN_QUANT_TYPE` 或 `VLLM_ILU_MLA_ATTN_QUANT_TYPE`），同时设 `VLLM_USE_V2_MODEL_RUNNER=0`。
- 开启 PP + MTP 兼容（设了 `VLLM_ILU_ENABLE_PP_MTP=1`）。

## 常用环境变量

### Model Runner V2



```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

该变量由用户设置：如需显式使用 V2，可设置为 `1`；如需强制回到 V1，可设置为 `0`。

使用量化 KV cache（即设了 `VLLM_ILU_FLASH_ATTN_QUANT_TYPE` 或 `VLLM_ILU_MLA_ATTN_QUANT_TYPE`）时，必须同时把该变量设置为 `0`。

### DeepSeek V4 示例变量



`VLLM_ILU_DS4_PREFILL_AS_DECODE` 默认是 `0`。设为 `1` 时，DeepSeek-V4 的 prefill 改走 decode sparse MLA，仅适用于 bf16 KV：

```bash
export VLLM_ILU_DS4_PREFILL_AS_DECODE=1
```

普通 Qwen3 快速验证通常不需要设置这个变量。

## 已清理的旧环境变量



`vllm-iluvatar` v0.27.1 中，部分旧量化环境变量已移除，改由 CT config 的 `iluvatar` 块或启动参数控制。新部署不应再依赖以下变量：

- `VLLM_MOE_W8A8_W4A8`（旧名 `VLLM_W8A8_MOE_USE_W4A8`）
- `VLLM_W4A8_FORMAT`
- `VLLM_W4A8_VERSION`
- `VLLM_W8A8_FORMAT`
- `VLLM_W8A8_GEMM_MODE`
- `ENFORCE_CUDA_GRAPH`

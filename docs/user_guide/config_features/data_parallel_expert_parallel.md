# 数据并行与专家并行



数据并行（DP）和专家并行（EP）用于在多卡、多副本或 MoE 模型场景下扩展推理服务能力。

## 适用场景



指南面向已支持 DP+EP 的 routed-expert 模型，并且只覆盖单机。

DP / EP 通常适合以下场景：

- 模型规模较大，需要跨多张加速卡部署。
- 请求量较高，希望通过数据并行扩展服务吞吐。
- 使用 MoE 模型，需要合理分布专家计算。
- 单纯 TP / PP 不能满足部署、显存或吞吐目标时，需要组合并行策略。

## 配置思路



部署前建议先确认以下信息（括号内是对应的启动参数或环境变量与取值示例，取值以本页下方的单机样例为例，按实际部署改）：

- 模型类型：dense 还是 MoE（dense 保持 `--data-parallel-size 1`；MoE 开专家并行 `--enable-expert-parallel`）。
- 模型精度：BF16、W8A8、W4A8 等（由权重自带的 `config.json` 量化配置决定，无单独启动参数）。
- 设备数量：单机多卡还是多机多卡（PP=1 时总卡数 = `--tensor-parallel-size` × `--data-parallel-size`，如 `4 × 4 = 16`）。
- 并行组合：TP、PP、DP、EP 的规模如何分配（如 `--tensor-parallel-size 4 --data-parallel-size 4 --enable-expert-parallel`，本例 `--pipeline-parallel-size` 不设即 PP=1）。
- 是否启用 CUDA Graph、MTP、量化 KV cache 等功能（CUDA Graph 用 `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}'`；MTP 用 `--speculative-config`；量化 KV cache 设 `VLLM_ILU_FLASH_ATTN_QUANT_TYPE` / `VLLM_ILU_MLA_ATTN_QUANT_TYPE`）。
- Model Runner 路径：是否需要显式指定 V1 / V2（`VLLM_USE_V2_MODEL_RUNNER=1` 用 V2、`=0` 回 V1）。

dense 模型和普通单服务保持 `--data-parallel-size 1`。

下面这条单机命令参考模型是 DeepSeek V4 W4A8，拓扑 TP=4、DP=4、PP=1。

```bash
MODEL=/path/to/DeepSeek-V4-w4a8

export VLLM_ILU_DP_ROUND_ROBIN_LB=1
export VLLM_ILU_DP_ENABLE_SYNC_PREFILLS=1
export VLLM_ILU_DP_PREFILL_MAX_DEFER_STEPS=8
export VLLM_ILU_DP_MAX_REPLACEMENT_WAIT_MS=5
export VLLM_ILU_DECODE_EP_BACKEND=deepep_low_latency
export VLLM_ILU_DISABLE_DEEPEP=1
export VLLM_ILU_DEEPEP_USE_INT8=1

vllm serve "${MODEL}" \
  --no-enable-prefix-caching \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 2048 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0,"max_cudagraph_capture_size":64}' \
  --served-model-name DeepSeek
```

本例 `VLLM_ILU_DISABLE_DEEPEP=1`，LL 使用 `BufferWrapper`/NVSM 路径，LL buffer 使用 `--max-num-seqs` 估算每个 rank 的最大 token 容量，因此启动时直接设置 `--max-num-seqs`，并让 CUDA Graph capture size 与其保持一致（本例都是 `64`）；修改 `--max-num-seqs` 后必须重新检查 LL buffer 和 CUDA Graph capture size。`VLLM_ILU_DISABLE_DEEPEP=1` 不表示关闭 EP 通信。

## 与 Model Runner V2 的关系



0.27.1 起使用 Model Runner V2；当前只支持下面这一种编译组合，其他组合（默认的 `VLLM_COMPILE` / `PIECEWISE` / `FULL` / `FULL_AND_PIECEWISE`）尚未适配。

- 如需显式使用 V2，可设置 `VLLM_USE_V2_MODEL_RUNNER=1`。
- 如需强制回到 V1，可设置 `VLLM_USE_V2_MODEL_RUNNER=0`。

## 与 CUDA Graph 的关系



v0.27.1 默认开启 CUDA Graph，当前只支持以下一种 compilation config（见 [CUDA Graph](cuda_graph.md)）：

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}'
```

如果 DP / EP 场景下遇到显存或启动问题，应优先检查 CUDA Graph 配置是否符合当前版本要求。

## 注意事项



- DP / EP 规模需要和实际设备数匹配，避免配置出的并行规模超过可用设备。
- 量化模型需要先确认权重已按 v0.27.1 新格式准备好。

## 弹性专家并行（Elastic EP）



本版支持弹性专家并行：在专家并行的基础上，服务运行期间可以在线扩大或缩小数据并行规模，不必重启。

需要使用 Ray 作为 data-parallel 后端，并开启 EPLB。关键启动参数：

```bash
vllm serve "${MODEL}" \
  --enable-expert-parallel \
  --enable-elastic-ep \
  --enable-eplb \
  --data-parallel-backend ray \
  --data-parallel-size 2 \
  --all2all-backend allgather_reducescatter \
  --tensor-parallel-size 1
```

运行时在线扩缩，向服务发请求调整数据并行规模（示例把 DP 调到 4）：

```bash
curl http://127.0.0.1:${PORT}/scale_elastic_ep \
  -H "Content-Type: application/json" \
  -d '{"new_data_parallel_size": 4}'
```

返回 HTTP 200 表示已接受，随后需要等待重配置稳定再继续发请求。

## 解码上下文并行（DCP）



本版支持解码上下文并行：在解码阶段把上下文切分到多张卡上，和张量并行、流水并行组合使用。用 `--decode-context-parallel-size` 指定规模。

```bash
# 示例（GQA 注意力）：TP=8，DCP=2
--tensor-parallel-size 8 --decode-context-parallel-size 2

# 示例（MLA 注意力）：TP=8，PP=2，DCP=8
--tensor-parallel-size 8 --pipeline-parallel-size 2 --decode-context-parallel-size 8
```

`--decode-context-parallel-size` 要和实际设备数、以及 TP / PP 的分配协调好。

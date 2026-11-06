# DSpark 投机解码



`vllm-iluvatar` v0.27.1 为 DeepSeek-V4 新增 DSpark 投机解码能力。该能力会强制使用 Model Runner V2。

## 启动示例



DeepSeek-V4-Flash W4A8 的最小 DSpark 启动示例如下：

```bash
vllm serve /path/to/DeepSeek-V4-Flash-W4A8 \
  --tensor-parallel-size 8 \
  --speculative-config '{"method":"dspark"}' \
  --served-model-name test
```

- draft 包含在 DeepSeek-V4 target 权重里，不需要再传 draft model。
- `method=dspark` 与 MTP 二选一，不要同时配。
- 完整并行、量化、CUDA Graph 组合按对应页，本页只给投机解码相关参数。

## 适用条件



- 本版 DSpark 投机解码用于 DeepSeek-V4。
- DeepSeek-V4 的 draft 在 target 权重里，不需要单独的 draft checkpoint。
- `method=dspark` 与 MTP 二选一。
- DSpark 强制 Model Runner V2。

## 与其它功能的关系



- DSpark 投机解码强制使用 Model Runner V2。
- 如果同时使用量化、DP / EP、CUDA Graph 等能力，各自的限制见[量化](quantization.md)、[数据并行与专家并行](data_parallel_expert_parallel.md)、[CUDA Graph](cuda_graph.md)。


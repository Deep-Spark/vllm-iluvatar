# 软件栈运行链路



本页说明 `vllm-iluvatar` 在 vLLM 和天数 CoreX 软件栈中的位置。用户启动服务时仍使用 `vllm serve` 命令；启动过程中，vLLM 加载 `vllm-iluvatar` 插件，插件再接入天数推理库和 CoreX runtime 完成硬件适配与推理执行。

## 调用链



以下链路聚焦 `vllm-iluvatar` 插件相关路径：

```text
用户命令（vllm serve / OpenAI HTTP API）
  ↓
天数 vLLM 本体
  ↓  entry_points 插件注册
vllm-iluvatar 插件
  ↓
ixformer
  ↓
算子库
  ↓
CoreX toolkit
  ↓
宿主机 driver 和 GPU
```

## 各层职责



| 层级 | 提供的能力 |
| --- | --- |
| 用户命令：`vllm serve` | 用户启动服务的入口，保持与 vLLM 生态一致的使用方式。 |
| 天数 vLLM 本体 | 提供服务框架、模型加载、请求调度和插件加载机制。 |
| `vllm-iluvatar` 插件 | 注册天数平台能力，承接 vLLM 请求，并把执行链路接到天数后端。 |
| `ixformer` | 位于插件和底层推理库之间，负责组织后端推理执行。 |
| 算子库 | 提供 attention、KV cache、GEMM/GEMV 等推理和算子能力。 |
| CoreX toolkit | 提供天数 GPU 的运行时、数学库、DNN 库和通信库。 |
| 宿主机 driver 和 GPU | 提供实际硬件和驱动能力，容器或运行环境需要能访问到它们。 |

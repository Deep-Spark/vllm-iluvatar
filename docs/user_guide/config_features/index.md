# 配置与功能



`vllm-iluvatar` v0.27.1 是转向 out-of-tree plugin 后的首个发布版本。插件单向依赖 vLLM，并与 vLLM 版本一一对应；本版本要求 `vllm>=0.27.1,<0.28.0`。

## 目录



| 页面 | 说明 |
| --- | --- |
| [启动参数与环境变量](launch_args_and_env.md) | 常用启动参数和少量调试环境变量。 |
| [量化](quantization.md) | 支持新版 `iluvatar-mixed` 量化格式；布局字段（如 `pack_version`）随量化组写入 checkpoint，W4A8 组当前为 `2`；旧 W4A8 / W8A8 权重需要重新离线量化。 |
| [数据并行与专家并行（DP / EP）](data_parallel_expert_parallel.md) | 面向大模型、多卡、多副本或 MoE 场景的并行部署能力。 |
| [CUDA Graph](cuda_graph.md) | v0.27.1 默认开启 CUDA Graph；当前只支持 `mode=0` 与 `FULL_DECODE_ONLY` 这一种组合。 |
| [Model Runner V2](model_runner_v2.md) | v0.27.1 起使用 Model Runner V2；部分场景（DSpark、量化 KV cache）对运行路径有强制要求。 |
| [投机解码（DSpark）](dspark_speculative_decoding.md) | DeepSeek-V4 新增 DSpark 投机解码能力，并强制使用 Model Runner V2。 |

模型支持范围见[支持模型](../supported_models.md)。v0.27.1 新增或扩展多类模型支持。

## 使用建议



- 如果使用 W4A8 / W8A8 量化模型，优先阅读量化章节，确认权重是否已按 v0.27.1 的新版规则重新量化。
- 如果部署 MoE 模型或多卡大模型，优先阅读数据并行与专家并行章节，并结合模型、设备数量和服务拓扑确认并行策略。
- 如果遇到 CUDA Graph、Model Runner V2、DSpark 相关问题，应先确认当前功能组合是否对运行路径有强制要求（DSpark 强制 V2，量化 KV cache 需 V1）。

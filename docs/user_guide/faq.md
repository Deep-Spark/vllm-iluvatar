# 常见问题



本页只做现象到已有正文的指引，不重写解决步骤。

| 现象 | 可能原因 | 去哪页 |
| --- | --- | --- |
| 服务起不来 / 自检失败 | CoreX runtime 文件缺失，或 `ixsmi` 异常 | [安装前确认](installation/index.md#env)、[安装自检](installation/installation_check.md) |
| 显存不足 OOM | 显存比例、并发序列数、上下文长度或批处理 token 数需要按部署目标调整 | [启动参数与环境变量：显存与并发](config_features/launch_args_and_env.md#memory-and-concurrency) |
| 精度不对 / 结果异常，疑似没有走量化 | 未量化 checkpoint 忘了 `--quantization ilu_runtime` 时，服务可以 READY，全程 BF16，没有任何报错 | [量化：运行时量化 `ilu_runtime`](config_features/quantization.md) |
| Model Runner V1 / V2 与量化 KV cache 冲突 | 量化 KV cache 不支持 V2，需要显式设置运行路径 | [Model Runner V2：量化 KV cache](config_features/model_runner_v2.md) |
| DP / EP 相关 OOM 或参数不匹配 | 并行规模、CUDA Graph 配置、Model Runner 路径或功能组合不匹配 | [数据并行与专家并行](config_features/data_parallel_expert_parallel.md) |
| CUDA Graph 相关启动 / 显存 / stream capture 问题 | 编译配置不符合本版要求，或需临时用 `--enforce-eager` 排除 | [CUDA Graph](config_features/cuda_graph.md) |
| 启动异常慢，或退出时报 process group 相关错 | 记录复现命令与完整日志，通过官方渠道反馈 | 本行即处理指引 |

其余通用报错遵循上游 vLLM，见 [vLLM 官方文档](https://docs.vllm.ai/)。

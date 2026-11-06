# vllm-iluvatar



`vllm-iluvatar` 是天数 GPU 的 vLLM 插件。它以 out-of-tree plugin 形式接入 vLLM，用户仍然用 `vllm serve` 启动服务，调用方式与 vLLM 生态保持一致。

## 它在软件栈里的位置



启动服务时，vLLM 本体通过 entry_points 加载 `vllm-iluvatar`，插件注册天数平台能力并承接请求，再把执行链路接到天数推理库（`ixformer` 及其依赖的算子库），由 CoreX runtime 和宿主机驱动完成硬件上的推理执行。完整调用链和各层职责见「用户指南 / 参考 / 软件栈运行链路」（左侧导航）。

在这套链路上，本软件栈提供模型支持、量化、数据并行与专家并行、CUDA Graph 与 Model Runner V2、投机解码等能力，逐项说明见[配置与功能](user_guide/config_features/index.md)。

## 这份文档用于什么



本文档用于在天数 GPU 上部署、验证和日常使用 vLLM 软件栈：怎么装、怎么起服务、支持哪些模型、有哪些参数和环境变量、版本之间怎么配套、出问题先查什么。

本指南不覆盖插件代码开发、新模型适配和发版流程，也不重复公版 vLLM 的基础概念、通用 API 和生态集成文档，这部分请查 vLLM 官方文档。

版本之间的配套关系见[版本说明](user_guide/reference/release_notes/index.md)。

## 从这里开始



| 你要做的事 | 去这里 |
| --- | --- |
| 第一次把服务跑起来 | [快速开始](user_guide/quickstart.md) |
| 装环境、确认装对了 | [安装](user_guide/installation/index.md) |
| 调参数、打开某项能力 | [配置与功能](user_guide/config_features/index.md) |
| 起不来、报错、结果不对 | [常见问题](user_guide/faq.md) |

更多按任务查找的入口见[用户指南](user_guide/index.md)。

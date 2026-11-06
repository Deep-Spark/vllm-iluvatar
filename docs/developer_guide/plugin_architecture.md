# 插件架构



vllm-iluvatar 通过 vLLM 官方插件入口接入运行时，目标是在不修改社区 vLLM 源码的前提下，把天数平台能力、运行期 patch、算子、模型、量化和 KV connector 注册到 vLLM。

## 两个 entry point



`pyproject.toml` 注册了两个入口：

- `vllm.platform_plugins` -> `vllm_iluvatar:register_platform`
- `vllm.general_plugins` -> `vllm_iluvatar:register_general`

`register_platform()` 在 `current_platform` 首次解析时执行，负责确认 `ixformer` 可用、打开 worker 启动补丁开关、应用早期兼容 shim、注入天数环境变量、注册 attention backend 和平台阶段 import hook，最后返回 `IluPlatform` 的类路径。

`register_general()` 在 vLLM 模块初始化完成后执行，负责应用晚期 general patch，然后注册 OOT CustomOp、模型、量化方法和 KV connector。它适合处理需要目标模块已经存在的运行期改写，不承担 import-time 兼容。

## 接入路径总览



| 路径 | 生效时机 | 作用 |
| --- | --- | --- |
| 平台 / 通信器 | `current_platform` 首次访问 | 返回 `IluPlatform`，并通过平台接口暴露天数通信器类路径 |
| Attention backend 覆盖 | `register_platform()` 或查询 backend 时 | 覆盖 vLLM 的 attention backend enum 到天数实现 |
| `vllm._custom_ops` 算子覆盖 | 目标模块导入时 | 通过 import hook 在模块加载后替换或补充 custom ops |
| 早期兼容 shim | 深层 `import vllm` 之前 | 补齐 import-time 依赖，避免 worker 进程过早失败 |
| `torch.ops._C` native ops | 插件 native 扩展导入时 | 通过 C++ extension 注册天数 native schema 和实现 |
| IR provider | 配置、初始化或 worker 阶段 | 让平台返回天数 IR kernel 入口和默认优先级 |
| distributed function override | general plugin 阶段 | 用晚期 patch 替换运行期分布式函数 |
| CustomOp OOT replacement | layer 构造时 | 通过官方 OOT 机制注册自定义 CustomOp 实现 |
| 量化方法注册 | general plugin 后、量化识别时 | 用官方 `@register_quantization_config` 注册多条量化线（如 `compressed-tensors` 由 `IluQuantConfig` 替换官方同名配置类，另有 `fp8`、`ilu_runtime`、`awq`/`gptq`、`mxfp8`、`modelopt_fp4` 等） |
| Model OOT replacement | 模型加载时 | 通过 `ModelRegistry` 注册天数模型实现 |
| KV connector 注册 | connector 创建前 | 把 `IluNixlConnector` / `IluNixlPullConnector` / `IluNixlPushConnector` 注册进 vLLM |

## 组件速览



- `IluPlatform` 继承社区 `CudaPlatform`，复用 CUDA-like 路径，并覆盖天数相关的 device capability、kernel import、IR provider、通信器、backend class 和配置更新。
- `IluCommunicator` 继承 `CudaCommunicator`，底层使用 `ixformer.distributed` collectives，并提供天数 All2All manager 选择。
- `vllm_iluvatar._C` 承载 native extension，配合 `torch.ops._C` schema 注册给上层 Python 路径使用。
- `vllm_iluvatar.kernels` 放置天数 kernel 侧实现入口，平台和 IR provider 会在初始化过程中导入相关能力。
- `IluNixlConnector` / `IluNixlPullConnector` / `IluNixlPushConnector` 通过官方 `KVConnectorFactory.register_connector` 显式注册，用于 KV transfer 场景。

算子、模型、量化、IR 和 KV transfer 的扩展细节不在本页展开。

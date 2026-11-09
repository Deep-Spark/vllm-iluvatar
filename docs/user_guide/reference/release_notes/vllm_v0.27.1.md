# vLLM v0.27.1 版本说明



vLLM 0.27.1 是天数 vLLM 转向 out-of-tree plugin 架构后的首个发布版本。x86_64 和 aarch64 均已发布。

## 基本信息



| 项目 | 内容 |
| --- | --- |
| 版本号 | vLLM v0.27.1 |
| 适用 CPU 架构 | x86_64；aarch64 |
| 版本配套 | `vllm-iluvatar` 与 vLLM 版本一一对应，本版锁定 `vllm>=0.27.1,<0.28.0` |

## 主要变化

### 插件架构



- 本版本是首个纯 out-of-tree plugin 版本。
- `vllm-iluvatar` 单向依赖 vLLM，并与 vLLM 主版本一一配对使用。
- 用户仍通过 `vllm serve` 启动服务，插件在启动过程中完成天数平台适配。

### 量化能力



- 量化格式升级为 `iluvatar-mixed`；布局字段（如 `pack_version`）随量化组写入 checkpoint，W4A8 组当前为 `2`。
- 旧版 W4A8 / W8A8 权重需要使用 `examples/iluvatar-compressed-tensors/quant.py`，配合新版 `configs` 和 `rules` 重新做离线量化。
- 本版提供覆盖 16 个模型族的 compressed-tensors rules/configs：DeepSeek R1/V3.1、DeepSeek V3.2、DeepSeek V4、GLM-5.2、Kimi-K2.5、MiniMax-M2.5/M2.7/M3、Qwen2、Qwen3、Qwen3.5、Qwen3.6-35B-A3B、Qwen3.6-MoE、Qwen3.8、Step-3.5/3.7。
- W4A8 / W8A8 的方法和布局选择改由 CT config 中的 `config_groups.<group>.iluvatar` 块描述，包括 `gemm_format`、`pack_version`、`gemm_mode` 和 `mixed_strategy`。
- 量化产物写入 checkpoint 的 `config.json`，加载时由插件自动读取，启动命令不再需要传递对应环境变量。
- 新增未量化 checkpoint 的在线量化通路 `ilu_runtime`。使用方式见[量化](../../config_features/quantization.md)页的运行时量化一节。

### 环境变量清理



以下环境变量已删除，相关配置改由 CT config 的 `iluvatar` 块或启动参数控制：

- `VLLM_MOE_W8A8_W4A8`：原 `VLLM_W8A8_MOE_USE_W4A8`，现由 `iluvatar` 块决定。
- `VLLM_W4A8_FORMAT` / `VLLM_W4A8_VERSION`
- `VLLM_W8A8_FORMAT` / `VLLM_W8A8_GEMM_MODE`
- `ENFORCE_CUDA_GRAPH`：CUDA graph 默认开启。

### 编译与 CUDA Graph



本版本默认开启 CUDA graph。目前仍需显式指定 `mode=0` 和 `FULL_DECODE_ONLY`：

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}'
```

### Model Runner V2



Model Runner V2 已可用：

- v0.27.1 起使用 Model Runner V2；如需显式指定，可设置 `VLLM_USE_V2_MODEL_RUNNER`（`1` 使用 V2，`0` 强制使用 V1）。
- DSpark 投机解码由插件强制使用 V2。
- 使用量化 KV cache（`VLLM_ILU_FLASH_ATTN_QUANT_TYPE` / `VLLM_ILU_MLA_ATTN_QUANT_TYPE`）时，需显式设置 `VLLM_USE_V2_MODEL_RUNNER=0`。

## 新增模型与指南



- DeepSeek-V4 新增 DSpark 投机解码支持。
- 新增 MiniMax-M2.5、MiniMax-M2.7、MiniMax-M3 支持。
- 新增 Qwen3.8（dense W8A8 / W4A8）支持；Flash Next 路径本版不支持。
- 同期提供 DP+EP 使用指南（见随版本发布的仓库内指南目录）。

## 使用注意事项



- `vllm-iluvatar` 插件依赖 vLLM，必须与对应 vLLM 版本配套安装和使用。
- 旧版 W4A8 / W8A8 权重不能直接沿用，需要按新版量化配置重新生成。
- 启用 CUDA graph 时，需按本页说明显式指定 `mode=0` 和 `FULL_DECODE_ONLY`。

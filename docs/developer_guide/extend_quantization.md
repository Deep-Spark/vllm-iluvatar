# 扩展量化



> 量化**怎么用**（怎么配、怎么启、注意事项）见「[用户指南/量化](../user_guide/config_features/quantization.md)」。本页写给要**扩展/提供量化**的开发者，只讲机制与扩展点骨架。

vllm-iluvatar 的量化扩展分三条线：权重量化走 compressed-tensors（guide 6）、KV Cache 量化走 attention 量化路径（guide 4）、不改磁盘的运行时量化走 `ilu_runtime`（guide 8）。下面各给扩展点骨架与边界。

## compressed-tensors 量化扩展



插件在 import 时用 `IluQuantConfig` 注册并替换官方 `compressed-tensors` 配置类；`IluQuantConfig` 继承 `CompressedTensorsConfig`，复用父类 scheme 选择逻辑，并把 config group 中的 `iluvatar` 扩展字段传给 Iluvatar 实现。

### 运行时如何选后端



`IluQuantConfig.get_quant_method(layer, prefix)` 对每个 Linear / MoE 层按固定顺序选后端：命中的 config group 含 `iluvatar.method` 时直接从 `IluvatarQuantRegistry` 实例化该类、跳过官方 scheme；否则按层类型判断——Linear 的 W8A8 INT8 / W4A8 INT 换成 Iluvatar Linear scheme，MoE（RoutedExperts）按 `weights`/`input_activations` 识别为 W4A8 / W8A8 MoE，其余保持官方 scheme。要点：共享专家通常是 Linear 而非 MoE；`iluvatar` 没有默认值，选用后要求字段必须显式填写；离线 `gemm_format` / `pack_version` 必须与运行时 `iluvatar` 一致。

### MoE / Linear 量化扩展点



- **MoE**：路由专家 config group 含非空 `iluvatar` 时走 Iluvatar MoE 后端；方案由官方 `weights`/`input_activations` 识别（4bit 权重 + 8bit 激活 → W4A8 MoE，动态 per-token W8A8 → W8A8 MoE），不使用额外开关字段。相关 `iluvatar` 字段：`gemm_format`（W4A8/W8A8 必填，`TN`/`NN`）、`pack_version`（W4A8 必填，`1`/`2`，当前 W4A8 MoE 实现使用 `2`）。
- **Linear**：与 MoE 同一扩展机制；config group 含 `iluvatar`、官方识别为 W8A8 INT8 / W4A8 INT 时换成 Iluvatar Linear 实现。各字段适用类型不同：`gemm_format`（W8A8 可 `TN`/`NN`；W4A8 Linear 仅 `NN`）、`weight_layout`（仅 W4A8，`packed_nn`）、`mixed_strategy`（仅 W8A8，`CHANNEL`）、`gemm_mode`（仅 W8A8，`NORMAL`/`PERF`）。字段必须与离线量化一致。

### 提供自定义 CT 量化格式



当官方 compressed-tensors 不存在某量化类型、或要在一份配置里混用官方无法组合的方案（如 Linear W8A8 + FP8 MoE）时，用 config group 的 `iluvatar.method` 指向 `IluvatarQuantRegistry` 中的类名；配置后跳过官方 scheme 选择、直接加载该类，`method` 在对应模型/量化模块 import 时注册，未注册的 `method` 加载时抛 `ValueError`。当前提供的 `method` 例如 `IluV4W8A16BMMLinearMethod`（DeepSeek-V4 `attn.wo_a`）、`IluGlmKVBProjW8A16Method`（GLM-5.2 `kv_b_proj`）、`IluFp8MoEMethod`（块缩放 FP8 路由专家）。离线量化走统一入口 `examples/iluvatar-compressed-tensors/quant.py`（configs 描述运行时配置、rules 描述离线转换规则）。

## KV Cache 量化扩展



Iluvatar 在社区 Half / BFloat16 / FP8 KV Cache 之外，补充面向天数硬件的 INT8 Attention 路径，启用由环境变量控制。

- **Full Attention（MHA/GQA）**：Q/K 动态 per-head 对称 INT8，V 可保持原始精度或静态 INT8。量化级别由 `VLLM_ILU_FLASH_ATTN_QUANT_TYPE` 控制（`KINT8` / `KVINT8`）；静态 V 需离线校准 scale，经 Attention Backend 内的 `StaticQuantManager` 采集（`VLLM_ILU_FLASH_ATTN_V_SCALE_PATH`，须 `--enforce-eager`、PP=1）。启用扩展点受限于 Full Attention（不支持 Cross Attention / sliding window / ALiBi / attention sinks / PCP·DCP>1）。
- **MLA（如 DeepSeek 系列）**：目前只支持 Per-Channel 量化，级别由 `VLLM_ILU_MLA_ATTN_QUANT_TYPE` 控制（`INT8`，Q 与 Latent KV 均动态量化，无需校准文件）。

## ilu_runtime 运行时量化扩展



它与 compressed-tensors（guide 6）职责不同，二者复用 `config_groups`/`targets`/`ignore`/`iluvatar.method` 的命名，但不是同一 schema，不能把一份配置直接复制到另一条路径。

两条路径的边界：

| | compressed-tensors（guide 6） | ilu_runtime（guide 8） |
| --- | --- | --- |
| 磁盘 | 生成或改写量化 checkpoint | 原 checkpoint 只读 |
| 配置位置 | HF `quantization_config.config_groups` | vLLM `additional_config.ilu_quant` |
| 顶层 `quant_method` | `compressed-tensors` | 保持原值（如 `fp8`） |
| `weights` 含义 | 磁盘已有的量化格式 | `source` / `target` / `loading` 描述加载前后两种格式 |
| method 入口 | 原有 constructor | 必须实现 `from_ilu_runtime_config(context)` |

### 配置入口 / 配置模型的扩展骨架



- **配置入口**：推荐 `vllm serve --config YAML`，`additional-config.ilu_quant` 下写 payload（也可 `--additional-config` 直传 JSON object）。`source` 必填（`inline` / `file`）。完全未量化的 checkpoint（无 `quantization_config`）没有挂载点，`ilu_quant` 会被静默忽略，须显式 `--quantization ilu_runtime`（`IluRuntimeQuantBaseConfig`：不读模型目录 json、checkpoint 只读、缺 `ilu_quant` 直接报错）。不要用注入假 `quantization_config` 的方式伪造量化身份。
- **配置模型**：顶层字段 `schema_version` / `enabled` / `applies_to`（`model_types` + `quant_methods`，后者填基础 quant 配置名）/ `ignore`；`config_groups` 每个 group 含 `priority` / `targets`（匹配运行时 module prefix）/ `require`（`original_disposition`、`checkpoint_dtypes`）/ `weights`（`source`/`target`/`loading`，`loading.mode` ∈ {`direct`,`transform`}、`loading.stage` ∈ {`none`,`process_weights_after_loading`,`weight_load`}）/ 激活 / `iluvatar.method`（要求实现 `from_ilu_runtime_config`）/ 可选 `validation`。
- **新增模型 method**：在 `IluvatarQuantRegistry` 注册 method 类并实现 `from_ilu_runtime_config(context)`，factory 须完整校验字段、不得把未知参数静默透传。当前参考实现：`IluFp8PerChannelOnlineLinearMethod`（Linear，`transform + process_weights_after_loading`）、`IluFp8ToFp4OnlineMoEMethod`（RoutedExperts，`transform + weight_load`）。示例按开源权重 ID 组织：`examples/iluvatar-runtime-quant/<org>/<model>/`，新增模型新建该目录、不平铺在根下。

<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 Iluvatar CoreX. All Rights Reserved.
-->
# Iluvatar 运行时量化使用指南

本文介绍如何在**不修改模型目录和 safetensors** 的前提下，通过
`additional_config.ilu_quant` 在加载时把 checkpoint 权重转换成 Iluvatar
运行时格式。当前已落地的首个 method 是 Qwen3.8-Flash-Next 的 BF16 dense
Linear → per-channel FP8。

| 项目 | 说明 |
| --- | --- |
| **适用对象** | 磁盘权重尚未完全量化、需要在加载时转换成 Iluvatar kernel 布局的模型 |
| **改动范围** | `vllm-iluvatar` 插件仓库。配置放在 vLLM `additional-config`，不写回模型 `config.json` |
| **参考实现** | `IluFp8PerChannelOnlineLinearMethod`；示例对齐 [Qwen/Qwen3.8-Flash-Next](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next) |

使用路径：

1. **推荐：YAML inline**。`--model` 指向原 checkpoint，`--config` 指向
   `examples/iluvatar-runtime-quant/<org>/<model>/` 下的 inline YAML。
2. **YAML + 外部 JSON**。JSON payload 放到所有 worker 的同一绝对路径，YAML
   只保留 `source: file`、路径和 SHA256，适合配置独立发布。
3. **派生目录对照**。只改隔离副本的 `config.json`，safetensors 链回原目录。
   见 [指南 6](6-how-to-use-and-provide-iluvatar-extend-compress-tensors.md) §4.2。

---

## 目录

1. [背景：为什么需要运行时量化](#1-背景为什么需要运行时量化)
2. [与指南 6 的边界](#2-与指南-6-的边界)
3. [配置入口](#3-配置入口)
4. [配置模型](#4-配置模型)
   - [顶层字段](#41-顶层字段)
   - [config_groups](#42-config_groups)
5. [示例目录约定](#5-示例目录约定)
6. [Qwen3.8-Flash-Next 示例](#6-qwen38-flash-next-示例)
   - [加载契约](#61-加载契约)
   - [三种配置模式](#62-三种配置模式)
   - [启动命令](#63-启动命令)
7. [故障定位](#7-故障定位)
8. [附录](#8-附录)

---

## 1. 背景：为什么需要运行时量化

官方 FP8 checkpoint 往往只把 routed experts 存成 block-FP8，并通过
`modules_to_not_convert`（旧配置也可能叫 `ignored_layers`）让普通 Linear
继续保存 BF16。要让这些 BF16 Linear 走 Iluvatar 的 per-channel FP8 kernel，
有两条路：

1. **离线转换**：生成一份新的量化权重目录。磁盘布局与运行时一一对应，但会
   多一份 checkpoint，后续官方权重更新也要再转一遍。
2. **运行时量化（本文）**：服务仍读取原 checkpoint；原 `weight_loader` 先完成
   TP / fused shard 装配，再在 `process_weights_after_loading` 里把本 rank
   的完整权重转成运行时格式。YAML/JSON 只是服务配置，不属于模型目录。

```text
checkpoint 权重
  → 原 vLLM weight_loader 完成 TP / fused 装配
  → process_weights_after_loading
  → Iluvatar 运行时权重
```

磁盘 safetensors 由 BF16 转成 FP8 是离线转换；读取 BF16 后在
`process_weights_after_loading` 中生成 FP8 是加载时转换。两者必须在字段、
日志和测试中明确区分。

运行时量化遵循：

1. **原 checkpoint 不可变**：不写入、不覆盖 `config.json`、safetensors、
   index、tokenizer；启动前后原目录的文件清单、大小和 SHA256 必须一致。
2. **fail-fast**：未知字段、非法组合、错误 SHA256、config group 冲突和计数
   不符都终止启动或首次推理，不静默回退到“看起来能跑”的缺省值。
3. **先保留原分发，再覆盖**：基础 quant config 先给出
   `original_method` / `original_disposition`；只有唯一 group 命中且前置条件
   满足时，才创建 runtime method。
4. **按本 rank shape 决策**：kernel 不支持的 `[N, K]` 完整回退
   `original_method`，不 padding、不先量化再反量化。
5. **注册表是能力边界**：只接受 `IluvatarQuantRegistry` 中的类名或显式别名，
   不接受任意 import path 或 callback。method 必须实现
   `from_ilu_runtime_config(context)`。
6. **forward 无配置查找**：分发时解析并缓存，forward 只执行选定 method。

当前 schema 可以描述 `direct` / `transform` 等加载方式，以及未来的 MoE、
Attention、INT8、FP4。真正已实现 runtime factory 的只有
`IluFp8PerChannelOnlineLinearMethod`（native BF16 → per-channel
`float8_e4m3fn`，`transform` @ `process_weights_after_loading`）。选择尚未
实现 factory 的注册 method 会明确报错。

---

## 2. 与指南 6 的边界

[指南 6](6-how-to-use-and-provide-iluvatar-extend-compress-tensors.md) 的主要对象是
已经离线量化、并把布局契约写入模型 `config.json` 的权重：

| | 指南 6（Compressed-Tensors 扩展） | 本文（runtime quant） |
| --- | --- | --- |
| 磁盘 | 生成或改写量化 checkpoint | 原 checkpoint 只读 |
| 配置位置 | HF `quantization_config.config_groups` | vLLM `additional_config.ilu_quant` |
| 顶层 `quant_method` | `compressed-tensors` | 保持原值（如 `fp8`） |
| `weights` 含义 | 磁盘已经具备的量化格式 | `source` / `target` / `loading` 描述加载前后两种格式 |
| method 入口 | 原有 constructor | 必须实现 `from_ilu_runtime_config(context)` |
| 示例 | `examples/iluvatar-compressed-tensors/` | `examples/iluvatar-runtime-quant/<org>/<model>/` |

二者复用 `config_groups`、`targets`、`ignore` 和 `iluvatar.method` 的命名，
但不是同一个 schema，也不能把一份配置直接复制到另一条路径。

PLE、routed experts 等已由原始 `IluFp8Config` 分发的层继续走原路径；
`additional_config.ilu_quant` 由插件独立 resolver 解析，不会让
`IluQuantConfig` 把本配置当成 Compressed-Tensors。

---

## 3. 配置入口

推荐使用 `vllm serve --config YAML`。vLLM 的 YAML 字段名使用 CLI 形式：

```yaml
additional-config:
  ilu_quant:
    source: inline
    # payload ...
```

也可在命令行直接传 JSON object：

```bash
vllm serve /path/to/model \
  --additional-config '{"ilu_quant":{"source":"inline", "...":"..."}}'
```

直接传 object 和 YAML inline 的语义相同。`ilu_quant` 不接受“JSON 文件路径”
字符串简写；引用文件必须使用显式的 `source: file` tagged union。

---

## 4. 配置模型

### 4.1 顶层字段

#### `source`

`source` 必填，只有两个值：

- `inline`：其余 payload 字段与 `source` 同级；
- `file`：wrapper 只允许 `source` / `file` / `sha256`。

file 模式要求：

- `file` 是绝对路径、普通 UTF-8 JSON 文件；
- 文件不超过 1 MiB；
- `sha256` 格式为 `sha256:<64 位小写十六进制>`；
- 每个 worker 先校验原始 bytes，再解析 JSON；
- JSON 从 `schema_version` 开始，不得再次包含 file wrapper 字段。

#### `schema_version` / `enabled`

- `schema_version` 当前只能为整数 `1`；
- `enabled` 必须是 boolean；为 `false` 时保留原 quant config 分发。

#### `applies_to`

`model_types` 和 `quant_methods` 都是非空、无重复字符串数组。当前模型必须
同时命中，否则报错，防止把一份配置误用到其他 checkpoint。

#### `ignore`

`ignore` 是全局排除数组，优先于所有 group。元素采用 vLLM Compressed-Tensors
target 语义：

- 完整运行时 module prefix；
- module 类名；
- `re:` 开头的正则；
- fused module 可通过基础 quant config 的 `packed_modules_mapping` 匹配。

空数组合法。非法正则在解析时失败。

### 4.2 config_groups

`config_groups` 是非空 object。每个 group 必须包含：

- `priority`：整数；多个 group 命中时选择唯一最高值，同最高值冲突报错；
- `targets`：非空 target 数组，匹配 **运行时 module prefix**，不是 checkpoint
  tensor 名称；
- `require`；
- `weights`；
- `input_activations`；
- `output_activations`；
- `iluvatar`；
- 可选 `validation`。

#### `require`

`original_disposition` 取值：

- `ignored`：原 quant config 明确将该层保持为非量化；
- `quantized`：原 quant config 原本会选择量化 method；
- `any`：不限制，只应由能够验证 source 格式的方法使用。

`checkpoint_dtypes` 是实际 checkpoint tensor 允许的 dtype。method 在参数创建
和真实 shard 加载时均校验，不能只相信 HF config。

#### `weights`

`weights.source` 描述磁盘权重：`format`（如 `native`、`fp8-block`）、
`dtypes`（必须与 `checkpoint_dtypes` 集合相同），以及其他由 method factory
校验的字段。

`weights.target` 描述运行时常驻权重：`format`、`dtype`、`strategy`；
group / block 等方案可由 method 要求额外字段。

`weights.loading.mode`：

- `direct`：磁盘权重直接作为运行时权重；
- `transform`：source 和 target 不同，加载后执行转换并释放 staging。

`weights.loading.stage`：

- `none`；
- `process_weights_after_loading`。

合法组合为：

```text
direct + none
direct + process_weights_after_loading
transform + process_weights_after_loading
```

`transform + none` 永远非法。当前只有
`IluFp8PerChannelOnlineLinearMethod` 实现 runtime factory，且只支持第三种
组合；schema 对 `direct` 的支持不代表已有 method 实现了它。

#### 激活

`input_activations` / `output_activations` 沿用
`num_bits` / `type` / `strategy` / `dynamic`。未使用时必须为 `null`。
具体组合由 method factory 校验。

#### `iluvatar`

`iluvatar.method` 是 `IluvatarQuantRegistry` 中的类名或显式别名。runtime
路径还要求该类实现：

```python
@classmethod
def from_ilu_runtime_config(cls, context):
    ...
```

其余字段都是 method-specific options。factory 必须完整校验，不能把未知参数
静默透传给 constructor。

#### `validation`

可选字段：`expected_selected`、`expected_transformed`、`expected_fallback`。
三者均按 worker / rank 统计，并满足：

```text
expected_selected = expected_transformed + expected_fallback
```

每个声明的 target 至少命中一次。超出期望立即失败；完整计数和 target coverage
在模型构造完成或首次 forward 前复核。

---

## 5. 示例目录约定

`examples/iluvatar-runtime-quant/` 按开源权重仓库 ID 组织，与 ModelScope /
Hugging Face 的 `<org>/<model>` 对齐，而不是自造 `qwen38-fp8` 这类缩写文件名。

官方权重 [Qwen/Qwen3.8-Flash-Next](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next)
对应：

```text
examples/iluvatar-runtime-quant/Qwen/Qwen3.8-Flash-Next/
├── fp8-inline.yaml      # 推荐：payload 全部写在 YAML
├── fp8-file.yaml        # source: file 包装，指向外部 JSON
└── fp8-runtime.json     # file 模式的 JSON payload
```

约定：

1. **目录名 = 模型 ID**。`<org>/<model>` 与开源权重命名一致，例如
   `Qwen/Qwen3.8-Flash-Next`、`deepseek-ai/DeepSeek-V3`。
2. **文件名区分配置形态和量化方案**。同一模型目录下用
   `<scheme>-inline.yaml` / `<scheme>-file.yaml` / `<scheme>-runtime.json`，
   当前 scheme 为 `fp8`。
3. **`--model` 指向实际 checkpoint**。示例路径只标识模型身份。本地目录可能
   带量化后缀（如 `Qwen3.8-Flash-Next-FP8`），或使用 modelscope / huggingface
   的 repo id；不要要求它和示例目录字符串完全相同。
4. **新增模型时新建 `<org>/<model>/` 目录**，不要把多个模型的 YAML 平铺在
   `iluvatar-runtime-quant/` 根下。

`fp8-file.yaml` 里的 `file` 必须是绝对路径。仓库中的值是占位路径，复制部署后
必须同时替换路径和 SHA256。

---

## 6. Qwen3.8-Flash-Next 示例

该配置只选择原 FP8 checkpoint 中被 `modules_to_not_convert` 标记为 BF16 的
dense Linear。routed experts 的 block-FP8、PLE n-gram FP8、embedding 和
`lm_head` 不被改写。

### 6.1 加载契约

```yaml
require:
  original_disposition: ignored
  checkpoint_dtypes: [bfloat16]
weights:
  source:
    format: native
    dtypes: [bfloat16]
  target:
    format: fp8
    dtype: float8_e4m3fn
    strategy: channel
  loading:
    mode: transform
    stage: process_weights_after_loading
input_activations:
  num_bits: 8
  type: float
  strategy: token
  dynamic: true
output_activations: null
iluvatar:
  method: IluFp8PerChannelOnlineLinearMethod
  gemm_format: TN
  shape_mismatch_policy: fallback_original
```

TP4 当前验收计数为：

```text
selected=242 transformed=158 fallback=84
```

TN kernel 在 TP / fused 切分后的本 rank `[N, K]` 上至少要求 `K % 64 == 0`；
GEMV 快路径还要求 `N % 32 == 0`。不满足时该层从 create / load / post-process
到 forward 都使用原 `UnquantizedLinearMethod` 和 BF16 权重。

### 6.2 三种配置模式

| 模式 | 模型路径 | 配置 | 原目录是否写入 |
| --- | --- | --- | --- |
| A 派生目录 | 隔离副本 | 只改副本 `config.json`，走既有 `IluFp8Config` 开关 | 否（只写副本） |
| B YAML inline（推荐） | 原 checkpoint | payload 全部写在 YAML | 否 |
| C YAML + 外部 JSON | 原 checkpoint | YAML 只保存文件身份和 SHA256 | 否 |

三种模式应选择相同的 242 个模块、使用同一 FP8 method，并产生 bit-exact 的
常驻 FP8 weight / scale 和 greedy token / logits。模式 A 适合作为兼容对照；
正式部署用 B 或 C。

### 6.3 启动命令

模式 B（推荐）：

```bash
vllm serve /data/nlp/Qwen/Qwen3.8-Flash-Next-FP8 \
  --config examples/iluvatar-runtime-quant/Qwen/Qwen3.8-Flash-Next/fp8-inline.yaml \
  --tensor-parallel-size 4
```

模式 C。先把 JSON 部署到所有 worker 的同一绝对路径，更新 wrapper 后再启动：

```bash
sha256sum examples/iluvatar-runtime-quant/Qwen/Qwen3.8-Flash-Next/fp8-runtime.json
vllm serve /data/nlp/Qwen/Qwen3.8-Flash-Next-FP8 \
  --config examples/iluvatar-runtime-quant/Qwen/Qwen3.8-Flash-Next/fp8-file.yaml \
  --tensor-parallel-size 4
```

模式 A 见指南 6 的派生目录生成器
`examples/iluvatar-compressed-tensors/qwen38_flash_next_fp8_online.py`。

---

## 7. 故障定位

- `does not apply to model_type/quant_method`：配置用于了错误模型或基础量化类型。
- `multiple ilu_quant groups`：多个命中 group 的最高 priority 相同。
- `requires original disposition ignored`：target 命中了原本已经量化的层，可能
  正在对 routed expert 二次量化。
- `expected checkpoint dtype`：实际 safetensors shard 不是声明的 BF16。
- `does not implement from_ilu_runtime_config`：该注册方法仅支持指南 6 的旧路径。
- `runtime quant counts ... do not match`：TP、模型版本、targets 或 shape
  capability 与生成 validation 数字时不同。
- `targets matched no layers`：把 checkpoint tensor 名称误当成了 runtime
  prefix，或模型实现的 fused module 名称已变化。
- `ilu_quant.file must be absolute` / `SHA256 mismatch`：file 模式未改成绝对
  路径，或 JSON 改过但未更新 SHA256。

---

## 8. 附录

### 8.1 关键文件

| 作用 | 文件 |
| --- | --- |
| runtime 配置解析与分发 | [`ilu_runtime_quant_config.py`](../../vllm_iluvatar/model_executor/layers/quantization/ilu_runtime_quant_config.py) |
| method 注册表 | [`ilu_quant_registry.py`](../../vllm_iluvatar/model_executor/layers/quantization/ilu_quant_registry.py) |
| FP8 基础 config 与 online Linear method | [`ilu_fp8.py`](../../vllm_iluvatar/model_executor/layers/quantization/ilu_fp8.py) |
| Qwen3.8-Flash-Next 示例 | [`examples/iluvatar-runtime-quant/Qwen/Qwen3.8-Flash-Next/`](../../examples/iluvatar-runtime-quant/Qwen/Qwen3.8-Flash-Next/) |
| 派生目录生成器（模式 A） | [`qwen38_flash_next_fp8_online.py`](../../examples/iluvatar-compressed-tensors/qwen38_flash_next_fp8_online.py) |

### 8.2 步骤速览

1. **确认 checkpoint**：顶层 `quant_method` 仍是原值（如 `fp8`）；候选层在磁盘上
   是 BF16，且被原 ignore 列表排除。
2. **按模型 ID 放置示例**：`examples/iluvatar-runtime-quant/<org>/<model>/`，
   与 ModelScope / Hugging Face 命名对齐。
3. **填写 `applies_to` 和 `targets`**：`targets` 用运行时 prefix；每个 target
   至少命中一次。
4. **选 mode B 或 C 启动**：inline YAML 最简单；配置要独立发布时用 file + SHA256。
5. **核对审计计数**：worker 日志中的 `selected` / `transformed` / `fallback`
   应与 `validation` 一致（Qwen TP4 为 242 / 158 / 84）。
6. **确认原目录未变**：启动前后 checkpoint 文件清单、大小和 SHA256 一致。

<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 Iluvatar CoreX. All Rights Reserved.
-->
# Iluvatar 运行时量化使用指南

本文介绍如何在**不修改模型目录和 safetensors** 的前提下，通过
`additional_config.ilu_quant` 在加载时把 checkpoint 权重转换成 Iluvatar
运行时格式。当前已落地的首个 method 是 BF16 dense Linear → per-channel FP8，
覆盖两类 checkpoint：部分量化的 Qwen3.8-Flash-Next-FP8，以及完全未量化的
Qwen3.8-27B。

| 项目 | 说明 |
| --- | --- |
| **适用对象** | 磁盘权重尚未完全量化、需要在加载时转换成 Iluvatar kernel 布局的模型 |
| **改动范围** | `vllm-iluvatar` 插件仓库。配置放在 vLLM `additional-config`，不写回模型 `config.json` |
| **参考实现** | `IluFp8PerChannelOnlineLinearMethod`；示例对齐 [Qwen/Qwen3.8-Flash-Next](https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next) 与 [Qwen/Qwen3.8-27B](https://modelscope.cn/models/Qwen/Qwen3.8-27B) |

使用路径：

1. **推荐：YAML inline**。`--model` 指向原 checkpoint，`--config` 指向
   `examples/iluvatar-runtime-quant/<org>/<model>/` 下的 inline YAML。
2. **YAML + 外部 JSON**。JSON payload 放到所有 worker 的同一绝对路径，YAML
   只保留 `source: file`、路径和 SHA256，适合配置独立发布。
3. **派生目录对照**。只改隔离副本的 `config.json`，safetensors 链回原目录。
   见 [指南 6](6-how-to-use-and-provide-iluvatar-extend-compress-tensors.md) §4.2。

checkpoint 本身**没有** `quantization_config` 时，还要显式加
`--quantization ilu_runtime`，见 [§3.1](#31-未量化-checkpoint-需要-quantization-ilu_runtime)。

---

## 目录

1. [背景：为什么需要运行时量化](#1-背景为什么需要运行时量化)
2. [与指南 6 的边界](#2-与指南-6-的边界)
3. [配置入口](#3-配置入口)
   - [未量化 checkpoint 需要 quantization ilu_runtime](#31-未量化-checkpoint-需要-quantization-ilu_runtime)
4. [配置模型](#4-配置模型)
   - [顶层字段](#41-顶层字段)
   - [config_groups](#42-config_groups)
5. [示例目录约定](#5-示例目录约定)
6. [Qwen3.8-Flash-Next 示例](#6-qwen38-flash-next-示例)
   - [加载契约](#61-加载契约)
   - [三种配置模式](#62-三种配置模式)
   - [启动命令](#63-启动命令)
7. [Qwen3.8-27B 示例：完全未量化的 checkpoint](#7-qwen38-27b-示例)
   - [加载契约](#71-加载契约)
   - [启动命令](#72-启动命令)
   - [实测计数与性能](#73-实测计数与性能)
8. [GLM-5.2 示例：MoE 的加载期 FP8 → FP4](#8-glm-52-示例moe-的加载期-fp8--fp4)
   - [为什么需要 weight_load](#81-为什么需要-weight_load)
   - [加载契约](#82-加载契约)
   - [启动命令](#83-启动命令)
   - [不支持的组合](#84-不支持的组合)
9. [故障定位](#9-故障定位)
10. [附录](#10-附录)

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

### 3.1 未量化 checkpoint 需要 quantization ilu_runtime

`ilu_quant` 由**基础量化配置**的 `get_quant_method` 解析。checkpoint 带
`quantization_config` 时（如 Qwen3.8-Flash-Next-FP8 的 `fp8`），vLLM 会自动构造
`IluFp8Config`，payload 自然被读到。

完全未量化的 checkpoint（如 Qwen3.8-27B，BF16、无 `quantization_config`）没有这个
挂载点：`ModelConfig.quantization` 是 `None`，vLLM 不会创建任何 quant config，
`get_quant_method` 不会被调用，`ilu_quant` 会被**静默忽略**——服务照常 READY、
全程 BF16、没有任何报错。

这类模型必须显式选择插件提供的空基础配置：

```bash
vllm serve /path/to/Qwen3.8-27B \
  --quantization ilu_runtime \
  --config examples/iluvatar-runtime-quant/Qwen/Qwen3.8-27B/fp8-inline.yaml
```

`ilu_runtime`（`IluRuntimeQuantBaseConfig`）的语义：

- 不读模型目录里的任何 json（`get_config_filenames()` 为空），checkpoint 只读；
- 每个 `LinearBase` 的原始分发固定是 `UnquantizedLinearMethod` +
  `original_disposition: ignored`，能改变它的只有命中的 ilu_quant group；
- 非 Linear 层返回 `None`，embedding / attention 与不加任何量化配置时一致；
- 没有 `ilu_quant`、或 `enabled: false` 时**直接报错**，不会退化成静默 BF16；
- 只能显式指定，不参与任何 checkpoint 自动识别。

不要用「给未量化 checkpoint 注入一份假的 `quantization_config`」来代替
（例如 `--hf-overrides '{"quantization_config": {"quant_method": "fp8", ...}}'`）。
那样 `ModelConfig.quantization` / `is_quantized` 会声称磁盘权重是 FP8，违反 §2 的
「原 checkpoint 量化身份不可伪造」，而且一旦 overrides 写错就退回静默 BF16。

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

`quant_methods` 填的是**基础量化配置**的名字，不是要转换成的目标格式：
checkpoint 自带 FP8 时是 `fp8`，完全未量化的 checkpoint 是 `ilu_runtime`
（§3.1）。

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

- `ignored`：原 quant config 明确将该层保持为非量化（`ilu_runtime` 下所有
  Linear 都属于这一类）；
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
- `process_weights_after_loading`：所有权重加载完成后整层转换；
- `weight_load`：**每个 checkpoint shard 落地即转换**。

合法组合为：

```text
direct + none
direct + process_weights_after_loading
transform + process_weights_after_loading
transform + weight_load
```

`transform + none` 和 `direct + weight_load` 永远非法（前者没有转换时机，
后者没有要转换的东西）。

两个 `transform` stage 的区别只有峰值显存，最终权重逐字节相同：

| | `process_weights_after_loading` | `weight_load` |
| --- | --- | --- |
| 转换时机 | 全部 shard 到齐后 | 每个 shard 到达时 |
| 峰值额外显存 | 整层 source + 整层 target | 一个 expert 的 source（MoE） |
| 实现难度 | 低，读现成的完整参数 | 高，要自己接管 `weight_loader` |
| 何时需要 | 默认 | 整层 source 装不下（大 MoE 的 TP 部署） |

已实现 runtime factory 的 method：

| method | 组合 | 层类型 |
| --- | --- | --- |
| `IluFp8PerChannelOnlineLinearMethod` | `transform + process_weights_after_loading` | Linear |
| `IluFp8ToFp4OnlineMoEMethod` | `transform + weight_load` | RoutedExperts |

schema 对 `direct` 的支持不代表已有 method 实现了它。

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
vllm serve /path/to/quantized-model \
  --config examples/iluvatar-runtime-quant/Qwen/Qwen3.8-Flash-Next/fp8-inline.yaml \
  --tensor-parallel-size 4
```

模式 C。先把 JSON 部署到所有 worker 的同一绝对路径，更新 wrapper 后再启动：

```bash
sha256sum examples/iluvatar-runtime-quant/Qwen/Qwen3.8-Flash-Next/fp8-runtime.json
vllm serve /path/to/quantized-model \
  --config examples/iluvatar-runtime-quant/Qwen/Qwen3.8-Flash-Next/fp8-file.yaml \
  --tensor-parallel-size 4
```

模式 A 见指南 6 的派生目录生成器
`examples/iluvatar-compressed-tensors/qwen38_flash_next_fp8_online.py`。

---

## 7. Qwen3.8-27B 示例

[Qwen/Qwen3.8-27B](https://modelscope.cn/models/Qwen/Qwen3.8-27B) 是 BF16 dense
权重，`config.json` 里没有 `quantization_config`，`hf_text_config.model_type` 是
`qwen3_5_text`。它与 §6 的差别：

| | Qwen3.8-Flash-Next-FP8 | Qwen3.8-27B |
| --- | --- | --- |
| 基础 quant config | `fp8`（checkpoint 自带） | `ilu_runtime`（必须显式指定） |
| 候选层来源 | `modules_to_not_convert` 里的 BF16 dense Linear | 整个语言塔 |
| targets | QSA indexer、PLE、MoE shared_expert | dense `mlp.gate_up_proj` / `down_proj` |
| 不量化 | routed experts、PLE n-gram、embedding、`lm_head` | ViT（`visual.*`）、embedding、`lm_head` |

### 7.1 加载契约

```yaml
applies_to:
  model_types: [qwen3_5_text]
  quant_methods: [ilu_runtime]
config_groups:
  qwen38_27b_bf16_to_fp8:
    priority: 100
    targets:
      - 're:.*\.self_attn\.qkv_proj$'
      - 're:.*\.self_attn\.o_proj$'
      - 're:.*\.linear_attn\.(?:in_proj_qkvz|in_proj_ba|out_proj)$'
      - 're:.*\.mlp\.(?:gate_up_proj|down_proj)$'
    require:
      original_disposition: ignored
      checkpoint_dtypes: [bfloat16]
```

`weights` / `input_activations` / `iluvatar` 与 §6.1 完全相同：native BF16 →
per-channel `float8_e4m3fn`，`transform` @ `process_weights_after_loading`，
动态 per-token FP8 激活，`gemm_format: TN`，`shape_mismatch_policy: fallback_original`。

`original_disposition` 只能是 `ignored`：checkpoint 没有声明任何量化，
`ilu_runtime` 给每个 Linear 的原始分发就是 `UnquantizedLinearMethod`。

### 7.2 启动命令

```bash
export IXFORMER_KERNEL_BACKEND=linear:ixinfer   # ViT 仍是 BF16
vllm serve /newdata/nlp/Qwen/Qwen3.8-27B \
  --quantization ilu_runtime \
  --config examples/iluvatar-runtime-quant/Qwen/Qwen3.8-27B/fp8-inline.yaml \
  --tensor-parallel-size 1 \
  --mamba-ssm-cache-dtype float32 \
  --hf-overrides '{"language_model_only":true}'
```

file 模式把 `--config` 换成 `fp8-file.yaml`，并先按 §4.1 更新绝对路径和 SHA256。

### 7.3 实测计数与性能

CoreX `4.5.0.20260722` / ivcore30 / vLLM 0.27.1，`--enforce-eager`，
`max_model_len=12288`，`max_num_seqs=32`：

```text
TP1   selected=304 transformed=304 fallback=0
TP2   selected=304 transformed=256 fallback=48
```

304 = 48 个 GDN 层 × 3 + 16 个 full-attention 层 × 2 + 64 个 MLP × 2。TP2 的 48 个
fallback 全部是 `linear_attn.in_proj_ba`：它的 `N = 2 * num_v_heads = 96`，TP2 切完
只剩 48，不满足 GEMV 的 `N % 32 == 0`，整层退回 BF16。示例 YAML 因此默认不写
`validation`——计数与 TP 绑定，固定 TP 部署时再取消注释。

常驻权重从 BF16 的约 53.8 GiB 降到约 30 GiB（24.4 GiB FP8 + BF16 的
embedding / `lm_head` / ViT）。加载峰值不是 BF16 全量：
`initialize_online_processing` 按层统计 `load_numel`，某层参数收齐立刻量化并释放
BF16 staging，额外峰值只有单层（最大 `gate_up_proj` 0.34 GiB）。

单请求 TTFT / ITL（`output_len=256`，与官方 block-FP8 权重同机同配置对照）：

| 配置 | TTFT 1k / 2k / 4k (ms) | ITL (ms) | decode (tok/s) |
| --- | --- | ---: | ---: |
| 27B BF16 + runtime quant，TP1 | 199 / 351 / 669 | 31.9 | 31.3 |
| 27B-FP8 官方 block-FP8，TP1 | 201 / 355 / 672 | 32.6 | 30.6 |
| 27B BF16 + runtime quant，TP2 | 179 / 327 / 628 | 34.5 | 29.0 |
| 27B-FP8 官方 block-FP8，TP2 | 142 / 255 / 486 | 29.9 | 33.5 |

TP1 两者持平（runtime quant 的 per-channel GEMM 略快于 block-FP8 的 requant 路径）。
TP2 慢约 15%，来源是那 48 个退回 BF16 的 `in_proj_ba`：每 token 多约 4.6 ms，
正好对应 48 次小 GEMM 的调度开销，而官方 block-FP8 权重里这些层仍是量化的。
需要 TP2 性能对齐时，可行方向是让只违反 GEMV 约束（`N % 32`）的层保持 FP8 并走
`fp8_per_channel_mm`，而不是整层回退。

单 rank 权重显存与官方 block-FP8 几乎相同（TP1 28.5 GiB vs 28.52 GiB，
TP2 均为 14.54 GiB），加载多花约 4 秒。同机纯 BF16 起服要 51.1 GiB，
`max_num_seqs=32` 时 GDN state + KV 已经放不下单卡，只能降到 4。

精度用 evalscope 1.11.1、`do_sample=false, temperature=0, top_p=1.0`，
与官方 block-FP8 权重同机同配置对照（`errored=0`）：

| 配置 | GSM8K (1319) | GPQA-Diamond (198) | GSM8K out tok/s | GPQA out tok/s |
| --- | ---: | ---: | ---: | ---: |
| 27B BF16 + runtime quant，TP1 | 0.9682 | 0.6919 | 9.36 | 11.09 |
| 27B-FP8 官方 block-FP8，TP1 | 0.9750 | 0.6515 | 9.29 | 10.83 |
| 27B BF16 + runtime quant，TP2 | 0.9704 | 0.7121 | 10.35 | 13.95 |
| 27B-FP8 官方 block-FP8，TP2 | 0.9750 | 0.6818 | 10.57 | 12.53 |

GSM8K 低 0.5～0.7 个点（1319 题里 9 题，约 1.4σ），GPQA 反而高 3～4 个点（198 题
的 1σ 就有 3.3 个点），两个方向都在噪声量级，没有系统性掉点。批量并发下的输出
吞吐两者相当——单请求 TP2 的 15% 差距在 batch 里被摊掉了。

## 8. GLM-5.2 示例：MoE 的加载期 FP8 → FP4

前两个示例都是 Linear 层的 `transform + process_weights_after_loading`。
这一个不同：它改的是 **routed experts**，而且用的是
`transform + weight_load`——**唯一因为显存放不下才必须用的 stage**。

### 8.1 为什么需要 `weight_load`

GLM-5.2 的 routed experts 是 block-FP8，今天走 `IluFp8MoEMethod`：
FP8 权重一直常驻，等 `process_weights_after_loading` 再整层转成 ix30 FP4。
FP4 只有 FP8 的约 1/4，所以**稳态完全放得下，放不下的是那一瞬间**——
同一时刻 FP8 原件和 FP4 结果都在显存里。TP8 因此起不来，只能用 16 卡。

`weight_load` 把转换提前到每个 shard 落地时：社区 loader 把 FP8 shard 写进一个
单 expert 的 staging（几 MiB），转换器直接写进常驻的 FP4 参数，staging 立刻释放。
峰值额外显存从"整层 FP8"降到"一个 expert 的 FP8"。

产出的四个参数与旧路径**逐字节相同**，
[`test_ilu_fp8_to_fp4_moe.py`](../../../tests/model_executor/layers/quantization/test_ilu_fp8_to_fp4_moe.py)
在 TP1 / TP2 两个 rank / EP 下都做了对拍，所以精度结论可以直接继承，
不需要重跑一遍评测。

### 8.2 加载契约

```yaml
weights:
  source:
    format: fp8-block
    dtypes: [float8_e4m3fn]
    block_shape: [128, 128]
  target:
    format: ix30-fp4
    dtype: uint8
    strategy: group
    group_size: 256
    scale_format: running_acc
  loading:
    mode: transform
    stage: weight_load
iluvatar:
  method: IluFp8ToFp4OnlineMoEMethod
  converter_backend: auto
  max_pending_shards: 8
```

与前两个示例的三处关键差异：

1. **`require.original_disposition` 是 `quantized` 而不是 `ignored`**。
   checkpoint 本来就被 `IluFp8Config` 认成 block-FP8 MoE，这里是替换它选中的
   method，不是给一个被 ignore 的层补量化。
2. **`targets` 指向 MoE 容器本身**（`re:.*\.mlp\.experts$`），不是里面的
   Linear。一个 target 覆盖 backbone 75 层 + MTP 1 层。
3. **`block_shape` 写在 `source.extra` 里**。它必须与 checkpoint
   `quantization_config.weight_block_size` 一致，不一致就报错——避免 checkpoint
   换块大小后被静默按旧值换算。

两个 method-specific option：

| option | 取值 | 含义 |
| --- | --- | --- |
| `converter_backend` | `auto`（默认）/ `torch` / `ixformer_native` | 直接透传给 ixformer registry 的 `backend=`。`torch` 是纯 PyTorch 参考实现，慢但确定；做等价性对拍时两边必须用同一个 backend |
| `max_pending_shards` | 正整数，默认 8 | 允许同时"只到了权重或只到了 scale"的 shard 数上限 |

`max_pending_shards` 是显存护栏而不是性能旋钮：每个半配对的 shard 都钉着一个
FP8 staging，堆积就等于回到了这条路径要避免的峰值。顺序 loader 下权重和 scale
紧邻到达，峰值只有个位数；超限说明 checkpoint 的到达顺序和假设不同，
报错会打印卡住的 `(group, local_expert, shard)`。

### 8.3 启动命令

checkpoint 自带 `quantization_config`，**不要**传 `--quantization`：

```bash
vllm serve /path/to/GLM-5.2-FP8 \
  --config examples/iluvatar-runtime-quant/ZhipuAI/GLM-5.2/fp4-moe-inline.yaml \
  --tensor-parallel-size 8 \
  --trust-remote-code
```

去掉 `--config` 就回到今天的行为。两条路径在代码里并存、互不引用，
`IluFp8MoEMethod` 的 compute 路径一行没改（新 method 继承它）。

### 8.4 不支持的组合

`create_weights` 里一律 fail-fast，不静默降级：带 bias 的 expert、
padding 过的 hidden dim 或 intermediate size、融合成 3D 的 expert 张量、
`is_transposed` 参数、BitsAndBytes、static input scale、EPLB、
以及非顺序的 `load_format`。

---

## 9. 故障定位

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
- `quantization='ilu_runtime' requires additional_config.ilu_quant`：选了
  `ilu_runtime` 却没给 payload。服务拒绝以静默 BF16 启动。
- **服务正常起来但一条量化日志都没有**：checkpoint 没有 `quantization_config`
  且忘了 `--quantization ilu_runtime`，`ilu_quant` 整块被跳过。见 §3.1。

`weight_load` stage 专有（§8）：

- `half-loaded expert shards exceed max_pending_shards`：权重和 scale 不是紧邻
  到达，staging 正在堆积。报错里列出了卡住的 `(group, local_expert, shard)`；
  先确认 `load_format` 是顺序 loader，再考虑调高上限——调高就等于接受更高的
  加载峰值。
- `expert shards were never converted to FP4`：某些 shard 只到了一半（通常是
  checkpoint 缺 `*_weight_scale_inv`，或 `make_expert_params_mapping` 没生成
  scale 条目）。此时参数里还是未初始化的字节，所以直接报错而不是放行。
- `requires a sequential weight loader`：`load_format` 可能并发喂 shard，会和
  staging 竞态。
- `does not support ...`（bias / padded hidden dim / EPLB / 3-D expert tensor）：
  见 §8.4，这些组合在 `create_weights` 就被拒绝。
- **转换很慢**：`converter_backend` 落到了 `torch`。ixformer 的 registry 在形状
  不满足 `n % 128 == 0 && k % 256 == 0` 时会自动回落 torch backend；用
  `IXFORMER_LOG_LEVEL` 看 registry 的选择。

---

## 10. 附录

### 10.1 关键文件

| 作用 | 文件 |
| --- | --- |
| runtime 配置解析与分发 | [`ilu_runtime_quant_config.py`](../../../vllm_iluvatar/model_executor/layers/quantization/ilu_runtime_quant_config.py) |
| 未量化 checkpoint 的基础 config | [`ilu_runtime_quant_base.py`](../../../vllm_iluvatar/model_executor/layers/quantization/ilu_runtime_quant_base.py) |
| method 注册表 | [`ilu_quant_registry.py`](../../../vllm_iluvatar/model_executor/layers/quantization/ilu_quant_registry.py) |
| FP8 基础 config 与 online Linear method | [`ilu_fp8.py`](../../../vllm_iluvatar/model_executor/layers/quantization/ilu_fp8.py) |
| FP8 source layout 与 ix30 FP4 转换入口 | [`ilu_fp8_to_fp4.py`](../../../vllm_iluvatar/model_executor/layers/quantization/ilu_fp8_to_fp4.py) |
| MoE 加载期 FP4 method | [`ilu_fp8_to_fp4_moe.py`](../../../vllm_iluvatar/model_executor/layers/quantization/ilu_fp8_to_fp4_moe.py) |
| 转换算子（双 backend，在 ixformer） | `ixformer/inference/functions/fp4/gemm/fp8_block_preprocess.py` |
| Qwen3.8-Flash-Next 示例 | [`examples/iluvatar-runtime-quant/Qwen/Qwen3.8-Flash-Next/`](../../../examples/iluvatar-runtime-quant/Qwen/Qwen3.8-Flash-Next/) |
| Qwen3.8-27B 示例（未量化 checkpoint） | [`examples/iluvatar-runtime-quant/Qwen/Qwen3.8-27B/`](../../../examples/iluvatar-runtime-quant/Qwen/Qwen3.8-27B/) |
| GLM-5.2 示例（MoE 加载期 FP4） | [`examples/iluvatar-runtime-quant/ZhipuAI/GLM-5.2/`](../../../examples/iluvatar-runtime-quant/ZhipuAI/GLM-5.2/) |
| 派生目录生成器（模式 A） | [`qwen38_flash_next_fp8_online.py`](../../../examples/iluvatar-compressed-tensors/qwen38_flash_next_fp8_online.py) |

### 10.2 步骤速览

1. **确认 checkpoint**：顶层 `quant_method` 仍是原值（如 `fp8`）；候选层在磁盘上
   是 BF16，且被原 ignore 列表排除。完全没有 `quantization_config` 时，改用
   `--quantization ilu_runtime`（§3.1）。
2. **按模型 ID 放置示例**：`examples/iluvatar-runtime-quant/<org>/<model>/`，
   与 ModelScope / Hugging Face 命名对齐。
3. **填写 `applies_to` 和 `targets`**：`targets` 用运行时 prefix；每个 target
   至少命中一次。
4. **选 mode B 或 C 启动**：inline YAML 最简单；配置要独立发布时用 file + SHA256。
5. **核对审计计数**：worker 日志中的 `selected` / `transformed` / `fallback`
   应与 `validation` 一致（Qwen TP4 为 242 / 158 / 84）。
6. **确认原目录未变**：启动前后 checkpoint 文件清单、大小和 SHA256 一致。

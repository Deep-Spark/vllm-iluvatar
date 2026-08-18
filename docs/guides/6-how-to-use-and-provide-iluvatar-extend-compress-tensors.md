<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 Iluvatar CoreX. All Rights Reserved.
-->
# Iluvatar 量化使用指南

本文介绍 Iluvatar 针对 MoE 和 Linear 层提供的量化优化。这些优化主要通过扩展 Compressed-Tensors 实现，用于适配已有量化方案或接入新的量化类型。对于新增或调整的量化类型，需要在插件中注册相应实现，并在运行前更新模型配置，从而完成特定模型的量化适配。

| 项目 | 说明 |
| --- | --- |
| **适用对象** | 所有 Iluvatar 或官方支持的量化类型 |
| **改动范围** | `vllm-iluvatar` 插件仓库；通过 Compressed-Tensors 扩展配置分发到 Iluvatar 特定量化后端 |

---

## 目录

1. [背景：量化](#1-背景量化)
2. [MoE 量化](#2-moe-量化)
3. [Linear 量化](#3-linear-量化)
4. [示例与参考](#4-示例与参考)
5. [附录](#5-附录)

---

## 1. 背景：量化

大语言模型的参数量和计算量较大。以 FP16/BF16 模型为例，权重、激活和 KV Cache 都会占用大量显存，Linear 与 MoE 层中的矩阵乘法也会消耗大量显存带宽和计算资源。量化将浮点数据映射为 INT8、INT4 等低比特数据，主要用于：

- 减少模型权重和运行时张量的显存占用，使更大的模型能够部署在单卡或较少的设备上；
- 降低权重读取与设备间通信的带宽开销；
- 利用硬件的低精度整数计算能力提升推理吞吐并降低时延。

量化通常需要在模型精度、显存占用和推理性能之间进行权衡。实际收益不仅取决于位宽，还取决于量化粒度、是否量化激活、量化参数的存储方式，以及硬件后端是否提供与该方案匹配的高效算子。

vLLM 对量化模型的支持大致可以分为以下两类：

1. **独立量化格式与后端**：AWQ、GPTQ 等较早且应用广泛的方案具有各自的模型配置、权重布局和执行后端。它们通常采用 W4A16 一类仅量化权重的方式：权重以 4 bit 保存，激活仍使用 FP16/BF16。此类方案可以显著降低权重显存和读取带宽，同时避免引入激活量化误差。在本文中，这类方案称为“非 Compressed-Tensors 量化”。
2. **Compressed-Tensors 描述的量化方案**：随着量化类型增加，仅依靠量化方法名称已不足以完整表达位宽、粒度、对称性和动态/静态量化等差异。vLLM 因此支持通过 Compressed-Tensors 配置描述更丰富的方案，例如 W8A8、W4A8 和 WNA16。W8A8/W4A8 同时压缩权重与激活，以进一步利用整数算力；WNA16 表示权重为 N bit、激活保持 16 bit，可覆盖不同权重位宽和分组策略。

> W、A 后的数字分别表示权重（Weight）和激活（Activation）的位宽。例如，W4A8 表示 4 bit 权重和 8 bit 激活。WNA16 中的 N 是权重位宽的占位符，具体值由量化配置决定。

Iluvatar 在 vLLM 现有 AWQ、GPTQ 和 Compressed-Tensors 支持的基础上，结合硬件的数据布局与整数计算能力，对 W8A8、W4A8 等方案增加了专用实现。对于官方已有格式，尽量复用其权重和配置；对于需要特定布局或算子的方案，则通过扩展 Compressed-Tensors 配置将模型分发到 Iluvatar 后端。

### 1.1 Compressed-Tensors 介绍

Compressed-Tensors 是一套声明式的压缩模型配置规范。它不只记录“量化方法名称”，而是分别描述权重、输入激活和输出激活的量化属性。vLLM 在加载模型时解析这些属性，并根据目标层和量化特征选择匹配的量化实现。

常用的量化描述字段如下。不同版本的 Compressed-Tensors 以及不同量化方案可使用的字段可能有所差异，未使用的字段通常省略或设置为 `null`。

| 字段 | 含义 | 常见值 |
| --- | --- | --- |
| `num_bits` | 量化位宽 | `4`、`8` |
| `type` | 量化数据类型 | `int`、`float` |
| `symmetric` | 是否使用对称量化 | `true`、`false` |
| `strategy` | scale/zero-point 的量化粒度 | `tensor`、`channel`、`group`、`block` |
| `group_size` | 分组量化时每组包含的元素数 | `128`、`-1`、`null` |
| `block_structure` | 块量化的块形状 | `[128, 128]`、`null` |
| `dynamic` | 量化参数是否在运行时动态计算 | `true`、`false` |
| `actorder` | 是否以及如何使用 activation ordering | `group`、`weight`、`null` |
| `scale_dtype` | scale 的存储类型 | `float32`、`null` |
| `zp_dtype` | zero-point 的存储类型 | `float32`、`null` |
| `observer` | 校准时使用的统计方法 | `minmax`、`memoryless` |
| `observer_kwargs` | observer 的附加参数 | `{}`、`null` |

一个量化配置组（config group）可包含以下三类描述：

- `weights`：权重量化方式；
- `input_activations`：输入激活量化方式；
- `output_activations`：输出激活量化方式；不量化时为 `null` 或省略。

并非每种方案都必须同时配置这三项。例如，WNA16 通常只需要描述 `weights`，而 W8A8/W4A8 还需要描述 `input_activations`。下面的配置片段表示：匹配 `Linear` 的层使用 8 bit 对称静态量化，输入激活采用 per-tensor 粒度，权重采用 per-channel 粒度。

```json
{
  "input_activations": {
    "dynamic": false,
    "num_bits": 8,
    "observer": "minmax",
    "observer_kwargs": {},
    "strategy": "tensor",
    "symmetric": true,
    "type": "int"
  },
  "output_activations": null,
  "targets": ["Linear"],
  "weights": {
    "dynamic": false,
    "num_bits": 8,
    "observer": "minmax",
    "observer_kwargs": {},
    "strategy": "channel",
    "symmetric": true,
    "type": "int"
  }
}
```

`targets` 指定该配置组匹配的层类型或层名称。一个模型可以定义多个配置组，使不同层采用不同的量化方式；模型加载时，各组会分别匹配对应的目标层。`ignore` 用于排除不适合量化的层，例如保留 `lm_head` 使用默认的非量化实现。

完整配置位于模型 `config.json` 的 `quantization_config` 中，结构示例如下：

```json
{
  "quantization_config": {
    "config_groups": {
      "group_0": {
        "targets": ["Linear"],
        "input_activations": {},
        "output_activations": null,
        "weights": {}
      },
      "group_1": {
        "targets": ["re:.*\\.experts"],
        "input_activations": {},
        "output_activations": null,
        "weights": {}
      }
    },
    "ignore": ["lm_head"]
  }
}
```

这里的 `{}` 仅用于省略具体量化属性；实际配置必须填写后端识别该量化方案所需的字段。配置只负责描述量化方式和目标层，权重文件本身还必须具有与该配置及执行后端一致的量化数据和布局，否则无法正确加载或计算。

## 2. MoE 量化

### 2.1 非 Compressed-Tensors 量化

对于 AWQ、GPTQ 等非 Compressed-Tensors 量化类型，可以直接使用与开源实现相同的权重数据。模型创建完成后，再按需对权重执行自定义处理；其使用方式与官方实现一致。

### 2.2 Compressed-Tensors 量化

对于 W8A8、W4A8 等通过 Compressed-Tensors 描述的量化类型，Iluvatar 支持多种权重数据布局，因此需要通过离线量化生成与目标算子匹配的布局。官方实现根据 Compressed-Tensors 配置选择后端；Iluvatar 在继承该机制的基础上进行扩展，以支持额外的数据布局和算子。

Iluvatar 在每个 config group 中增加扩展字段 `iluvatar`，用于配置量化实现的选择及其后端参数。目前，MoE config group 支持以下字段：

| 字段 | 适用类型 | 值 |
| --- | --- | --- |
| `gemm_format` | W4A8 / W8A8 | `TN` / `NN` |
| `pack_version` | W4A8 | `1` / `2` |
| `wna16_use_w4a8` | W4A8 | `true` / `false` |

各实现会校验自身所需字段及其取值范围；缺少必填字段或使用非法值可能导致模型加载失败。请确保在 `iluvatar` 字段中提供所选后端需要的全部参数。

## 3. Linear 量化

### 3.1 非 Compressed-Tensors 量化

与 MoE 量化类似，AWQ、GPTQ 等方案的使用方式与官方实现一致。

### 3.2 Compressed-Tensors 量化

Linear 量化与 MoE 量化使用相同的扩展机制。Linear config group 支持以下字段：

| 字段 | 适用类型 | 值 |
| --- | --- | --- |
| `gemm_format` | W4A8 / W8A8 | W4A8 使用 `NN`；W8A8 使用 `TN` / `NN` |
| `mixed_strategy` | W8A8 | `CHANNEL` / `TENSOR` |
| `gemm_mode` | W8A8 | `NORMAL` / `PERF` |

同样，各实现会校验自身所需字段及其取值范围。

### 3.3 特殊量化

除 3.1 和 3.2 节涉及的字段外，目前还提供实验性字段 `method`，用于选择 Iluvatar 特定量化实现。配置该字段后，会跳过官方 scheme 的选择流程，并根据 `method` 加载对应的实现类。

#### 3.3.1 特殊类型接入

该机制的目的之一，在于接入官方实现中不存在的量化类型，避免每次新增量化类型时重复修改分发逻辑。`method` 目前主要面向特定模型，通过 [`IluvatarQuantRegistry`](../../vllm_iluvatar/model_executor/layers/quantization/ilu_quant_registry.py) 注册。相关方法在对应的模型模块中注册，并随模型模块加载。目前提供的方法包括：

- `IluV4W8A16BMMLinearMethod`
- `IluGlmKVBProjW8A16Method`

#### 3.3.2 混合量化扩展

目前 vLLM 官方不支持混用非 Compressed-Tensors 与 Compressed-Tensors 描述的量化类型。通过 `IluvatarQuantRegistry` 注册实现并配置转换规则，可以在 Iluvatar 平台上支持 W8A8 + FP8 MoE 等混合量化方案。

## 4. 示例与参考

### 4.1 标准介绍

`quant.py` 是统一的离线量化入口。它将运行时量化配置与离线权重转换规则拆分为两个外部文件，避免不同模型的量化脚本重复实现权重分类、反量化、再量化和配置生成逻辑。具体量化算法及其注册表位于 `examples/iluvatar-compressed-tensors/methods.py`；`quant.py` 只负责规则解析、任务调度和模型文件 I/O。

#### 4.1.1 configs：运行时量化配置

`examples/iluvatar-compressed-tensors/configs/*.json` 是直接写入模型 `config.json` 中 `quantization_config` 的 Compressed-Tensors 配置，主要描述：

- Compressed-Tensors 的 `config_groups`、`targets` 和权重/激活量化字段；
- Iluvatar 的 `gemm_format`、`pack_version`、`method` 等扩展字段。

configs 不包含离线转换方法、checkpoint 名称匹配规则或模型配置 patch。后端量化方法仍按照 `get_quant_method -> get_scheme -> _get_scheme_from_parts` 的流程进行选择。`IluQuantConfig` 继承自 `CompressedTensorsConfig`，复用父类的 scheme 选择逻辑，并将 config group 中的扩展字段传递给 Iluvatar 优化实现。

对于 `weights`、`input_activations`、`output_activations` 这三类描述，建议从 rules 下的对应文件中引用相关参数，以保证离线量化参数与运行时配置一致。例如：

**config.json：**

```json
{
  "weights": {
    "num_bits": 8,
    "type": "int",
    "symmetric": {
      "$rules": "rules.dense_fp8_to_w8a8.quant.options.symmetric"
    },
    "strategy": "channel",
    "group_size": {
      "$rules": "rules.dense_fp8_to_w8a8.quant.options.group_size",
      "none_as": -1
    },
    "dynamic": false
  }
}
```

**rules 文件：**

```json
{
  "quant": {
    "method": "int8_per_channel",
    "options": {
      "group_size": -1,
      "symmetric": true,
      "gemm_format": "TN",
      "round_through_bfloat16": true
    }
  }
}
```

对于 `iluvatar` 字段，同样建议引用 rules 中的对应参数，以确保配置一致：

**config.json：**

```json
{
  "iluvatar": {
    "gemm_format": {
      "$rules": "rules.dense_fp8_to_w8a8.quant.options.gemm_format"
    },
    "mixed_strategy": {
      "$rules": "rules.dense_fp8_to_w8a8.iluvatar.mixed_strategy"
    },
    "gemm_mode": {
      "$rules": "rules.dense_fp8_to_w8a8.iluvatar.gemm_mode"
    }
  }
}
```

**rules 文件：**

```json
{
  "quant": {
    "method": "int8_per_channel",
    "options": {
      "group_size": -1,
      "symmetric": true,
      "gemm_format": "TN",
      "round_through_bfloat16": true
    }
  },
  "iluvatar": {
    "mixed_strategy": "CHANNEL",
    "gemm_mode": "PERF",
    "method": "IluV4W8A16BMMLinearMethod",
    "bmm_w8a16_gemm_format": "k16n16"
  }
}
```

由于离线权重转换不涉及输入、输出激活量化，`input_activations` 和 `output_activations` 如有需要，应直接在 config 文件中配置。

#### 4.1.2 rules：离线转换规则

`examples/iluvatar-compressed-tensors/rules/*.json` 按 checkpoint tensor 名称选择离线转换 pipeline。每条规则主要包含：

- `priority`：当一个权重同时匹配多条规则时，优先使用数值较大的规则；
- `action`：`drop` 表示丢弃且不写入转换后的权重，`keep` 表示原样保存，`convert` 表示按规则转换后保存；
- `match`：使用 `regex` 接收名称、`exclude_regex` 排除名称，并可附加 `suffixes`、`dtypes` 和 `ndims` 条件；
- `dequant`：指定源权重的反量化 `method`，`options` 为该方法接收的参数；
- `quant`：指定目标权重的量化 `method` 及其 `options`；
- `scale`：`from_suffix` 为源权重后缀，`to_suffix` 为 companion scale 后缀，用于根据权重名称定位对应的 scale；
- `outputs`：`weight`、`weight_scale`、`i8_weight_scale`、`i8_weight_zero` 等输出后缀；`weight` 可用于将 `weight_packed` 等源名称改写为运行时使用的 `.weight`；
- `work_device`：为 `cuda` 时，在执行量化方法前将反量化结果移动到命令行指定的工作设备。

当前 `methods.py` 提供以下反量化方法：

- `identity`：不改变输入权重；
- `to_float32`：将 BF16、FP16 等浮点权重转换为 FP32；
- `fp8_block`：使用普通浮点 block scale 还原 FP8 权重；
  - `options`：`block_size`。
- `fp8_block_e8m0`：使用 UE8M0 block scale 还原 FP8 权重；
  - `options`：`block_size`。
- `fp8_static`：使用可广播的静态 scale 还原 FP8 权重；
  - `options`：`block_size`、`allow_unscaled`。
- `mxfp4_block32`：解包 FP4 E2M1，并使用 UE8M0/32 scale 还原权重；
- `int4_group`：解包 Compressed-Tensors `weight_packed` INT4 权重，并使用 per-group scale 还原权重。
  - `options`：`num_bits`、`group_size`。

当前提供以下量化方法：

- `identity`：保持反量化结果的数据类型与布局；
- `cast`：将结果转换为指定浮点类型；
  - `options`：`dtype`。
- `int8_per_channel`：按输出通道执行对称 INT8 量化，并生成对应 scale；
  - `options`：`round_through_bfloat16` 控制是否先转换为 BF16 进行截断；`transport` 为 `true` 时交换输入 tensor 的第 0、1 维。
- `fp8_per_channel`：使用每个输出通道的绝对最大值执行对称 FP8 量化，并生成 FP32 scale；
  - `options`：`dtype` 默认为 `float8_e4m3fn`；`round_through_bfloat16` 和 `transport` 含义与 `int8_per_channel` 相同。
- `int4_moe`：先生成 per-channel INT8，再通过 ixformer 重排为 MoE INT4，并按版本生成 scale、`i8_weight_scale` 和 `i8_weight_zero`。
  - `options`：`group_size`、`symmetric`、`gemm_format`、`pack_version`、`scale_layout` 和 `transport`；当 `scale_layout` 为 `row` 时，scale 会 reshape 为 `(1, *)`。
- `pack_int4_moe`：处理三维专家权重，即所有专家位于同一个 tensor 中的情况；该方法会逐个专家调用 `int4_moe`，`options` 包括 `int4_moe` 参数外，提供：`packed`: gate_up 为 True, down 为 False；`chunk_dim` gate_up 的拆分维度。

对于不同的量化方法，需要同步调整 configs 和 rules，以保证离线量化参数与运行时配置一致。除了手动同步，还可以在 configs 中通过 `$rules` 引用 rules 字段。这样，修改离线量化参数后，运行时配置会自动更新。例如：

```json
{
  "iluvatar": {
    "group_size": {
      "$rules": "rules.id.quant.options.group_size",
      "none_as": -1
    },
    "gemm_format": {
      "$rules": "rules.id.quant.options.gemm_format"
    }
  }
}
```

#### 4.1.3 量化流程总览

```text
伪代码如下：
  for weight, name in all_weights:
    rule = match(name)
    if rule.drop:
      continue
    elif rule.keep:
      save(
        weight=weight
      )
    elif rule.convert:
      scale = rule(weight, name, from_suffix, to_suffix)
      dequant = rule.dequant(weight, scale)
      quant_results = rule.quant(dequant)
      save(
        weight = rule.output(quant_results, name),
        weight_scale = rule.output(quant_results, name),
        ...
      )
```

#### 4.1.4 约定

1. 顶层 `quant_method` 使用 `iluvatar-compressed-tensors`，`format` 使用 `iluvatar-mixed`。
2. 对于量化为 W4A8/W8A8 的非 MoE Linear，应在对应 config group 中设置 `format: int-quantized`，并提供该实现需要的参数。
3. 所有传递给 `iluvatar` 后端量化方法的参数，统一放在每个 group 的 `iluvatar` 字段中。
4. 4.1.2 节提到的反量化和量化算法均注册在 `examples/iluvatar-compressed-tensors/methods.py` 中。适配新模型时应优先复用已注册算法；确实无法满足需求时，再新增算法并注册。
5. configs 中的 `targets` 匹配 vLLM 运行时模块前缀，rules 中的 `match` 匹配 checkpoint tensor 名称。两者所处阶段不同，不应直接复制彼此的正则表达式。

#### 4.1.5 MTP 约定

当前示例对 MTP 层和非 MTP 层采用相同的量化方式。若 MTP 层需要独立的精度、布局或运行时实现，应分别调整 rules 和 configs。

#### 4.1.6 多模态部分约定

如果模型包含 visual 等模块，示例默认不对其进行量化；如有需要，请自行调整。

### 4.2 新模型适配检查清单

1. **检查源模型**：确认 `config.json`、权重索引、tensor 名称、dtype、shape 及 companion scale 后缀。大型索引应通过定向搜索或流式统计分析，避免整体打印。
2. **划分权重类别**：区分路由专家、共享专家、dense Linear、Attention、norm、router、embedding 和输出头等。
3. **编写 rules**：使用 checkpoint tensor 名称编写匹配条件；优先添加 dtype 约束和精确后缀，避免已转换权重被重复转换。
4. **复用转换算法**：优先使用 `methods.py` 中已有方法；新增方法时，应先使用小型张量或单个真实 tensor 验证反量化结果。
5. **编写 configs**：使用 vLLM 运行时模块前缀编写 `targets`。例如，专家通常匹配 `re:.*\\.experts`，不要直接复制包含专家编号或 `w1/w2/w3` 的 checkpoint 正则表达式。
6. **同步参数**：通过 `$rules` 对齐 `group_size`、`symmetric`、`gemm_format`、`pack_version` 等离线和运行时参数。
7. **执行小规模验证**：先验证 JSON、规则加载和匹配统计，再对少量 shard 或代表性 tensor 进行转换。
8. **验证输出**：检查输出 tensor 名称、dtype、shape、scale 后缀、权重索引和最终 `quantization_config`。
9. **启动推理**：使用目标并行配置加载模型，并至少执行一次短输入推理以验证后端选择和数值有效性。

### 4.3 模型示例：基于 DeepSeek-V4-Flash

#### 4.3.1 rules

完整内容见 `examples/iluvatar-compressed-tensors/rules/deepseek_v4.json`。

该文件定义以下处理方式：

1. 默认行为：`keep`，用于未命中其他规则的 norm、bias、BF16 权重等；
2. 共享专家：`fp8_block_e8m0` 反量化后执行 `int8_per_channel`；
3. 路由专家：`mxfp4_block32` 反量化后执行 `int4_moe`；
4. Attention 及相关 dense 权重：`fp8_block_e8m0` 反量化后执行 `int8_per_channel`；
5. 其余 FP8 权重：`fp8_block_e8m0` 反量化后转换为 BF16；
6. 已消费的源 scale：`drop`。

第 5 类规则可以匹配第 2、4 类中的 FP8 权重，但其默认优先级最低，因此不会覆盖更具体的规则。如需让这些权重保留为 BF16，可以提高第 5 类规则的优先级。路由专家的源权重为 MXFP4，不受该 FP8 fallback 规则影响。

#### 4.3.2 configs

完整内容见 `examples/iluvatar-compressed-tensors/configs/deepseek_v4.json`。

该文件指定以下运行时量化方式：

1. 共享专家：W8A8；
2. 路由专家：W4A8；
3. `attn.wo_a`：使用 `IluV4W8A16BMMLinearMethod`；
4. 其他 Attention 及相关 dense 层：W8A8；
5. 其他层：使用 vLLM 默认的非量化后端。

#### 4.3.3 转换前验证

```bash
python3 examples/iluvatar-compressed-tensors/quant.py \
  --input /path/to/DeepSeek-V4-flash-0731 \
  --configs examples/iluvatar-compressed-tensors/configs/deepseek_v4.json \
  --rules examples/iluvatar-compressed-tensors/rules/deepseek_v4.json \
  --match-only \
  --match-report examples/iluvatar-compressed-tensors/deepseek_v4_match.json
```

该命令不会执行实际转换，而是根据 configs 和 rules 进行匹配，并将各规则的匹配结果写入 `match-report` 文件，用于转换前的快速验证。

#### 4.3.4 转换

```bash
python3 examples/iluvatar-compressed-tensors/quant.py \
  --input /path/to/DeepSeek-V4-flash-0731 \
  --output /path/to/DeepSeek-V4-flash-0731-quant \
  --configs examples/iluvatar-compressed-tensors/configs/deepseek_v4.json \
  --rules examples/iluvatar-compressed-tensors/rules/deepseek_v4.json
```

#### 4.3.5 运行

```bash
# 无需设置额外环境变量；后端根据 Compressed-Tensors 配置中的字段进行选择。
vllm serve /path/to/DeepSeek-V4-flash-0731-quant \
  -tp 2 \
  -pp 4 \
  --enforce-eager
```

## 5. 附录

### 5.1 文件职责速查

| 文件 | 主要职责 | 不应包含 |
| --- | --- | --- |
| `configs/<model>.json` | 描述运行时量化组、目标模块、权重/激活精度及 Iluvatar 扩展字段 | checkpoint tensor 匹配规则、离线反量化方法 |
| `rules/<model>.json` | 按 checkpoint tensor 名称选择 `keep`、`drop` 或 `convert`，并定义离线转换参数 | 运行时实现类的选择逻辑 |
| `methods.py` | 注册可复用的反量化和量化算法 | 模型专用 tensor 名称匹配 |
| `quant.py` | 解析配置和规则、遍历 shard、转换权重、重建索引并写入模型配置 | 模型专用量化算法 |

configs 与 rules 应使用相同的文件名 stem，例如 `configs/deepseek_v4.json` 对应 `rules/deepseek_v4.json`。

rules 文件还包含以下顶层字段：

| 字段 | 说明 |
| --- | --- |
| `schema_version` | rules 文件格式版本，当前为 `1` |
| `name` | 规则集名称，通常与文件名 stem 一致 |
| `load_device` | shard 加载设备；`requested` 使用命令行指定设备，`cpu` 强制从 CPU 加载 |
| `model_config_patch` | 写出模型配置前应用的 patch；键支持点分路径，值为 `null` 时删除对应字段 |
| `default_action` | 权重未命中任何规则时执行的动作，可选 `keep` 或 `drop` |

### 5.2 `$rules` 引用

rules 是离线量化参数的事实来源。configs 可以通过 `$rules` 引用具体规则中的字段，以避免离线参数和运行时参数不一致：

```json
{
  "iluvatar": {
    "group_size": {
      "$rules": "rules.routed_mxfp4_to_w4a8.quant.options.group_size",
      "none_as": -1
    }
  }
}
```

引用路径格式为 `rules.<rule_id>.<field...>`。`none_as` 用于处理离线参数与运行时配置的表示差异：当 rules 中的值等于 `none_as` 时，写入模型配置的值为 `null`。例如，离线算法以 `-1` 表示 per-channel，而 Compressed-Tensors 运行时配置使用 `null`。

注意：

- 具体参数应写在 rules 中，configs 通过 `$rules` 引用；不要在 rules 中反向使用 `$config`。
- 被引用的 rule ID 和字段必须存在，否则量化任务会在读取配置时失败。

### 5.3 命令行参数

| 参数 | 说明 |
| --- | --- |
| `--input` | 输入模型目录；必须包含 `model.safetensors.index.json` 和 safetensors shard |
| `--output` | 输出模型目录 |
| `--configs` | 运行时量化配置文件 |
| `--rules` | 离线转换规则文件 |
| `--device` | 工作设备，可选 `cpu` 或 `cuda`，默认为 `cuda` |
| `--shard-start` | 可选，选择 shard 切片的起始下标 |
| `--shard-end` | 可选，选择 shard 切片的结束下标，不包含该下标 |
| `--match-only` | 仅根据索引中的 tensor 元数据统计规则匹配结果，不加载或转换 tensor |
| `--match-report` | 将规则匹配结果写入指定 JSON 文件；需与 `--match-only` 一同使用 |

只有完整转换全部输入 shard 时，脚本才会重建 `model.safetensors.index.json` 并写入最终 `config.json`。分片任务的输出在完成合并和最终索引生成前不能直接用于推理。

### 5.4 常见问题

#### 量化运行报错

- 方法不存在：检查方法是否已在 `methods.py` 中注册，或特殊量化实现是否已在 `IluvatarQuantRegistry` 中注册。
- OOM：运行时设置 `--device cpu`；对于需要在命令行指定设备上执行量化的算法，在对应 rule 中设置 `"work_device": cuda`。

#### 规则未命中预期权重

- 检查 rules 匹配的是 checkpoint tensor 名称，而不是 vLLM 运行时模块名。
- 检查 `regex`、`suffixes`、`dtypes` 和 `ndims` 是否同时满足。
- 当多条规则都能匹配时，确认 `priority` 是否符合预期。
- 查看转换结束时输出的规则统计，确认每类权重的命中数量。

#### 运行时无法选择或选择了错误的量化实现

- 检查 configs 的 `targets` 是否匹配 vLLM 构造后的模块前缀，而不是原始 checkpoint tensor 名称。
- 检查 `format`、权重/激活量化字段以及 Iluvatar 扩展字段是否与目标 scheme 一致。
- 对特殊实现，确认 `iluvatar.method` 对应的实现已注册到 `IluvatarQuantRegistry`。

#### 加载时找不到 weight / scale

- 检查是否选择了正确的量化实现。
- 检查 `scale.from_suffix` 是否确实是当前权重名称的后缀。
- 检查 `scale.to_suffix` 是否与索引中的 scale 名称一致，例如 `.scale`、`.weight_scale` 或 `.weight_scale_inv`。

#### 权重能够加载但推理结果异常

- 确认离线转换与运行时配置中的 `group_size`、`symmetric`、`gemm_format` 和 `pack_version` 完全一致。
- 检查 scale 的布局、dtype 和广播方向。
- 确认源权重使用了正确的反量化方法，尤其要区分普通 FP8 block scale、UE8M0 scale、静态 scale 和已打包 INT4。
- 使用单个 tensor 做“反量化 → 再量化”验证，再进行完整模型转换。

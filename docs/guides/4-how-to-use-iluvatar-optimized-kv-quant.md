<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright (c) 2026 Iluvatar CoreX. All Rights Reserved.
-->
# Iluvatar KV Cache 量化使用指南

## Attention Quant

本节介绍 Iluvatar 在 Attention 中提供的额外 INT8 KV Cache 量化能力，目前支持 FlashAttention 和 FlashMLA 后端。后端通过 platform 自动选择调用，对模型使用无感。由于官方上游目前不支持该类量化，因此需要在运行前对少量上游代码进行 Patch，但该过程对模型开发和使用透明。

| 项目 | 说明 |
| --- | --- |
| **适用对象** | 所有 Iluvatar 或官方支持的、包含**标准 Attention** 或 **MLA（Multi-head Latent Attention）** 的模型 |
| **改动范围** | `vllm-iluvatar` 插件仓库；对少量官方代码进行 Patch，用于创建和管理 KV Cache scale |

---

## 目录

1. [背景：为什么需要 Attention 量化](#1-背景为什么需要-attention-量化)
2. [Full Attention 量化支持](#2-full-attention-的量化支持)
3. [MLA 量化支持](#3-mla-量化支持)
4. [如何量化 Full Attention](#4-如何量化-full-attention)
5. [示例与参考](#5-示例与参考)
6. [附录](#6-附录)

---

## 1. 背景：为什么需要 Attention 量化

### vLLM 开源社区的量化优化方式

大模型推理中的 Attention 不仅包含 Q/K/V 相关计算，还需要为每个请求持续保存 KV Cache。随着 batch size 和上下文长度增加，KV Cache 会快速增长，并逐渐成为显存容量、访存带宽和服务并发能力的主要瓶颈。因此，Attention 量化的收益同时体现在计算和存储两方面。

vLLM 社区对 KV 量化的实现方式主要通过 FP8 KV Cache，支持 E4M3/E5M2、Per-tensor 或部分后端下的 Per-head scale；scale 可使用默认值、在 warmup 阶段动态计算，或通过 `llm-compressor` 使用校准数据集离线生成。FP8 KV Cache 主要用于降低显存占用和访存带宽，部分后端还会将 Q 一并量化并直接执行 FP8 Attention。

FP8 的实际可用范围和加速效果取决于硬件及 Attention 后端，Iluvatar 在支持社区 Half、BFloat16 以及 FP8 KV Cache 的基础上，补充了面向天数硬件的 INT8 Attention 路径：标准 Attention 支持动态 INT8 Q/K 与可选的静态 INT8 V，MLA 支持动态 INT8 Q 和 Latent KV，从而进一步发挥天数硬件的 INT8 算力与带宽优势。

### Iluvatar KV INT8 量化支持

- 在 prefill 阶段，将 Q/K 等计算转换为 INT8，可以利用更高的 INT8 算力和更低的数据搬运开销，降低首 Token 时延（FTL）。
- 在 decode 阶段，每一步都需要读取历史 KV Cache。使用 INT8 KV Cache 可将其主体数据的显存占用降至 FP16/BF16 的约一半，同时减少显存带宽压力，从而支持更长上下文和更高并发，并提升吞吐。
- KV Cache 占用降低后，同一设备可分配更多 cache block，减少因显存不足造成的请求排队或抢占；该收益通常会随上下文长度和并发数增加而更加明显。

---

## 2. Full Attention 的量化支持

本文所称的标准 Full Attention，是指 vLLM 中常见的 MHA（Multi-Head Attention）和 GQA（Grouped-Query Attention）自注意力：每个 Token 可以关注其可见范围内的全部历史 Token，并分别维护 K Cache 和 V Cache。它区别于 MLA、Sliding Window Attention 和 Cross Attention 等结构。

- 使用量化时，Q/K 采用动态 Per-head 对称 INT8 量化，每个 Head 的 scale 在推理过程中实时计算；V 则可以保持原始精度，也可以采用静态 INT8 量化。
- 采用静态 INT8 V 时，需要额外提供预先生成的 scale，并在推理过程中使用该 scale 完成 V 的量化与反量化。由于当前实现使用固定 scale，推理期间不会对其进行动态更新，而固定 scale 与不同输入下实时计算的最优 scale 可能存在偏差，因此需要通过具有代表性的校准数据集进行离线后训练量化（PTQ）。该过程与 vLLM 社区使用 `llm-compressor` 为 FP8 KV Cache 离线生成 scale 类似：在覆盖充分且贴近实际业务的数据上统计合适的量化范围，以兼顾推理精度，并避免运行时动态计算 V scale 的额外开销和实现复杂度。

### 2.1 量化级别

使用环境变量 `VLLM_ILU_FLASH_ATTN_QUANT_TYPE` 控制量化级别：

| 级别 | 含义 | Q/K | V | 是否需要 Scale 文件 | KV Cache 布局 | 相对 KV Cache 占用 |
|------|------|-----|---|----------------|---------------|-------------------|
| **None**<sup>A</sup> | 无量化（默认） | 原始精度 | 原始精度 | 否 | `[B, 2, H, D]`<sup>B</sup> | 100%<sup>C</sup> |
| **KVINT8** | INT8 Q/K + **INT8 V** | 动态量化 | **静态量化** | **是** | `[B, 2, H, D]` + `[B, H]` + `[H, D]`<sup>D</sup> | 约 50%<sup>E</sup> |
| **KINT8** | INT8 Q/K + 原始精度 V | 动态量化 | 原始精度 | 否 | `[B, H, D]` + `[B, H]` + `[B, H, D]`<sup>F</sup> | 约 75%<sup>G</sup> |

---

**KVINT8** 的显存优化最高，但需要使用校准数据集生成 V 的静态 scale。
**KINT8** 的显存优化低于 `KVINT8`，适用于无法进行数据集校准或对 V 的量化精度存在顾虑的场景。

<small>
<sup>A</sup> None 表示无需设置该环境变量。
<sup>B</sup> 为便于描述，表中布局按数据存入 KV Cache 前的序列形式表示。
<sup>C</sup> 100% 表示原始 KV Cache 的占用基线。
<sup>D</sup> B 为 batch 维度，H 为 num_kv_heads，D 为 head_dim。三部分依次为 INT8 K/V Cache、K 的 float32 scale 和 V 的 float32 scale。
<sup>E</sup> 相对占用约为 (2 × B × H × D × sizeof(int8) + B × H × sizeof(float32)) / (2 × B × H × D × sizeof(原始类型))。V scale 的占用很小，计算中忽略；当原始类型为 fp16，约为 50%。
<sup>F</sup> 三部分依次为 INT8 K Cache、K 的 float32 scale 和原始精度 V Cache。
<sup>G</sup> 忽略 K scale 的较小额外开销时，当原始类型为 fp16 时，INT8 K 与原始精度 V 分别约占基线的 25% 和 50%，合计约为 75%。
</small>

### 2.2 使用限制

Full Attention 的 INT8 KV Cache 量化在使用时有以下约束，将在后续逐步去除：

| 约束 | 说明 |
|------|------|
| 仅支持 Full Attention | Cross Attention 及其它类型 Attention 变体暂不支持 |
| 不支持 Sliding window | 要求 `sliding_window == (-1, -1)` |
| 不支持 ALiBi | 要求 `alibi_slopes is None` |
| 不支持 attention sinks | 要求 `sinks is None` |
| 不支持 PCP/DCP > 1 | 要求 `PCP / DCP == 1` |

---

## 3. MLA 量化支持

对于 MLA（Multi-head Latent Attention，例如 DeepSeek 系列），由于其独特设计，不使用与 Full Attention 完全相同的量化方式。目前，MLA 量化只支持 Per-Channel<sup>A</sup> 量化。

### 3.1 量化级别

使用环境变量 `VLLM_ILU_MLA_ATTN_QUANT_TYPE` 控制量化级别：

| 级别 | 含义 | Q | Latent KV | 是否需要校准文件 | KV Cache 布局 | 相对 KV Cache 占用 |
|------|------|-----|---|----------------|---------------|-------------------|
| **None**<sup>B</sup> | 无量化（默认） | 原始精度 | 原始精度 | 否 | `[B, D]` | 100%<sup>C</sup> |
| **INT8** | INT8 Q + **INT8 Latent KV** | 动态量化 | 动态量化 | 否 | `[B, D]` + `[B, 2]`<sup>D</sup> | 约 50%<sup>E</sup> |

---

<small>
<sup>A</sup> Per-Channel 分别针对 non-RoPE 和 RoPE 部分进行量化。
<sup>B</sup> None 表示无需设置该环境变量。
<sup>C</sup> 100% 表示原始 KV Cache 的占用基线。
<sup>D</sup> B 为 batch 维度，D 为 latent head_dim。两部分依次为 INT8 Latent Cache、Cache 的 float32 scale。
<sup>E</sup> 相对占用约为 (B × D × sizeof(int8) + B × 2 × sizeof(float32)) / (B × D × sizeof(原始类型))。当原始类型为 fp16 时，约为 50%。
</small>

---

## 4. 如何量化 Full Attention

vLLM 社区通常使用 `llm-compressor` 基于校准数据集统计 scale。Iluvatar 则通过环境变量开启 scale 采集逻辑：启动 vLLM 服务后，使用校准数据集发送推理请求，即可完成 V scale 的统计与导出。

### 4.1 原理简介

Iluvatar 的 Attention Backend 中集成了 `StaticQuantManager` 模块。将环境变量 `VLLM_ILU_FLASH_ATTN_V_SCALE_PATH` 设置为有效的输出文件路径后，模型会在推理过程中持续采集并更新 V scale，达到指定步数时将结果写入该文件。

### 4.2 启动 vLLM 服务

```bash
export VLLM_ILU_FLASH_ATTN_V_SCALE_PATH="/path/to/v_cache_scales.pt"
# Set other vLLM-supported options as needed.

vllm serve /Path/to/model \
  -tp 2 \
  --enforce-eager \
  ...
```

`VLLM_ILU_FLASH_ATTN_V_SCALE_PATH` 的父目录必须已存在，且目标文件不能预先存在。
**PP** 需设置为 1；受当前实现限制，暂不支持在 PP > 1 时采集 scale。
必须设置 **`--enforce-eager`**。CUDA Graph 会跳过模型推理中的部分 Python 执行路径，而 `StaticQuantManager` 依赖该路径完成 scale 采集。

### 4.3 发送校准请求

可以使用任意 vLLM 支持的请求格式发送校准数据，推荐使用 OpenAI 兼容接口。简单示例如下，完整示例请参考 5.2 节：

```bash
curl 127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"/Path/to/model",
    "messages":[{"role": "user", "content": "This is an example input"}],
    "max_tokens":1024}'
```

### 4.4 检查校准结果

可通过以下任一方式确认校准过程正常进行：

1. 观察服务端打印日志：

   ```text
   StaticQuantManager save to /path/to/v_cache_scales.pt with 200 step
   ```

2. 检查 `VLLM_ILU_FLASH_ATTN_V_SCALE_PATH` 指定的文件是否已经生成：`ls -l "${VLLM_ILU_FLASH_ATTN_V_SCALE_PATH}"`。

当前实现会在模型前向推理达到一定次数后写入文件，默认每 100 步保存一次。请确保校准数据和请求数量充足；确认结果文件已生成后，即可停止 vLLM 服务。

### 4.5 注意事项

- 该量化仅适用于包含 Full Attention 的模型；DeepSeek 系列模型使用 MLA，设置该参数无效。
- 暂不支持仅对指定层进行量化，后续版本可能调整该行为。
- 使用 draft 模型时，如果 draft 模型包含 Full Attention，默认也会一并进行 scale 采集；后续版本可能调整该行为。

---

## 5. 示例与参考

### 5.1 Full Attention 使用 KINT8

- **本小节使用 Qwen3-8B**

#### 5.1.1 启动 vLLM 服务

```bash
export VLLM_ILU_FLASH_ATTN_QUANT_TYPE="KINT8"
vllm serve Qwen/Qwen3-8B \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}
```

#### 5.1.2 简单测试请求

```bash
curl 127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen3-8B",
    "messages":[{"role": "user", "content": "who are you?"}],
    "temperature":0.0,
    "max_tokens":256}'
```

#### 5.1.3 精度测试

```bash
# 使用 mmlu 进行简单精度测试，其它测试请自行验证
evalscope eval \
  --model Qwen/Qwen3-8B \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets mmlu  \
  --stream \
  --timeout 6000 \
  --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":20480}' \
  --eval-batch-size 32
```

### 5.2 量化并使用 KVINT8

- **本小节使用 Qwen3.6-35B-A3B**，该模型结合 Full Attention、Mamba 和 MTP。

#### 5.2.1 启动 vLLM 服务

```bash
export VLLM_ILU_FLASH_ATTN_V_SCALE_PATH="/path/to/model/v_scales.pt"

vllm serve Qwen/Qwen3.6-35B-A3B \
    -tp 4 \
    --max-num-seqs 64 \
    --enforce-eager \
    --skip-mm-profiling \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}' \
    --reasoning-parser qwen3
```

#### 5.2.2 发送请求

- 在 mmlu 的 abstract_algebra 子集上进行校准，其它数据集和类型请自行设计。

```bash
evalscope eval \
  --model Qwen/Qwen3.6-35B-A3B \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets mmlu  \
  --dataset-args '{"mmlu":{"subset_list":["abstract_algebra"]}}' \
  --stream \
  --timeout 6000 \
  --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":20480, "extra_body":{"chat_template_kwargs":{"thinking": true}}}' \
  --eval-batch-size 32

# 预期结果示例：
# ┌─────────────────┬───────────┬──────────┬──────────────────┬───────┬─────────┬─────────┐
# │ Model           │ Dataset   │ Metric   │ Subset           │   Num │   Score │ Cat.0   │
# ├─────────────────┼───────────┼──────────┼──────────────────┼───────┼─────────┼─────────┤
# │ Qwen3.6-35B-A3B │ mmlu      │ mean_acc │ abstract_algebra │   100 │    0.97 │ STEM    │
# └─────────────────┴───────────┴──────────┴──────────────────┴───────┴─────────┴─────────┘
```

#### 5.2.3 重启 vLLM 服务

```bash
export VLLM_ILU_FLASH_ATTN_V_SCALE_PATH="/path/to/model/v_scales.pt"
export VLLM_ILU_FLASH_ATTN_QUANT_TYPE="KVINT8"

vllm serve Qwen/Qwen3.6-35B-A3B \
    -tp 4 \
    --max-num-seqs 64 \
    --skip-mm-profiling \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}' \
    --reasoning-parser qwen3 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}
```

#### 5.2.4 精度测试

```bash
evalscope eval \
  --model Qwen/Qwen3.6-35B-A3B \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets mmlu  \
  --dataset-args '{"mmlu":{"subset_list":["abstract_algebra"]}}' \
  --stream \
  --timeout 6000 \
  --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":20480, "extra_body":{"chat_template_kwargs":{"thinking": true}}}' \
  --eval-batch-size 32
# Score 应仍为 0.97
```

### 5.3 MLA 使用 INT8

- **本小节使用 GLM-5.2**

#### 5.3.1 启动 vLLM 服务

```bash
export VLLM_ILU_MLA_ATTN_QUANT_TYPE="INT8"

vllm serve ZhipuAI/GLM-5.2 \
    -tp 4 \
    -pp 4 \
    --max-num-seqs 64 \
    --enforce-eager
```

#### 5.3.2 简单测试请求

```bash
curl 127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"ZhipuAI/GLM-5.2",
    "messages":[{"role": "user", "content": "who are you?"}],
    "temperature":0.0,
    "max_tokens":256}'
```

#### 5.3.3 精度测试

```bash
# 使用 mmlu 进行简单精度测试，其它测试请自行验证
evalscope eval \
  --model ZhipuAI/GLM-5.2 \
  --api-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets mmlu \
  --stream \
  --timeout 6000 \
  --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":20480, "extra_body":{"chat_template_kwargs":{"thinking": true}}}' \
  --eval-batch-size 32
```

### 5.4 故障排查

- **Error**：`The total kv heads(config.json) must > 0, but get *`。当 `Num_kv_heads < TP` 时，部分 TP rank 的 `v_scale` 相同，保存时需要清除冗余 scale。该逻辑需要完整的 `Num_kv_heads` 和 TP 信息，因此会在初始化时尝试获取。由于不同模型在 `config.json` 中定义 `Num_kv_heads` 的字段名可能不同，构建 `StaticQuantManager` 时会先尝试从常见字段中获取；如果获取失败，则从环境变量 `VLLM_ILU_FLASH_ATTN_TOTAL_KV_HEADS` 中获取，该环境变量的默认值为 `-1`。出现此错误时，请从 `config.json` 中获取正确的 `Num_kv_heads` 值，并通过该环境变量提供；否则，保存的 scale 信息可能有误。

## 6. 附录

### 关键文件清单

| 作用 | 文件 |
| --- | --- |
| StaticQuantManager 功能提供 | [`vllm-iluvatar/vllm_iluvatar/v1/attention/ilu_static_quant.py`](../../vllm_iluvatar/v1/attention/ilu_static_quant.py) |
| FlashAttention Backend 实现 | [`vllm-iluvatar/vllm_iluvatar/v1/attention/backends/ilu_flash_attn.py`](../../vllm_iluvatar/v1/attention/backends/ilu_flash_attn.py) |
| MLA Backend 实现 | [`vllm-iluvatar/vllm_iluvatar/v1/attention/backends/mla/ilu_mla.py`](../../vllm_iluvatar/v1/attention/backends/mla/ilu_mla.py) |

---

### 步骤速览

1. **确定量化类型**：包含 `Full Attention` 或 `MLA` 的模型可以使用量化。使用 `KVINT8` 需要进行数据集校准，`KINT8` 或 `INT8` 可直接使用。
2. **KVINT8 校准**：设置 `VLLM_ILU_FLASH_ATTN_V_SCALE_PATH`，启动服务并发送足量请求，完成量化。
3. **KVINT8 使用**：设置 `VLLM_ILU_FLASH_ATTN_QUANT_TYPE` 和 `VLLM_ILU_FLASH_ATTN_V_SCALE_PATH`，切换为量化模式。

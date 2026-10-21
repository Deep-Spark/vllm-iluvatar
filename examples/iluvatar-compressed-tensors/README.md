<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd. -->

# Checkpoint quantization

`configs/` 与 `rules/` 按模型厂商/生态分组，两个目录中的配方使用相同的相对路径。
`--recipe <vendor>/<name>` 自动选择对应的两个 JSON，名称不带 `.json`。
路径以 `quant.py` 所在目录为基准，执行命令时不需要切到特定工作目录。

| 目录 | 模型系列 |
| --- | --- |
| `deepseek/` | DeepSeek R1、V3.2、V4 |
| `minimax/` | MiniMax M2.5、M2.7、M3 |
| `moonshot/` | Kimi K2.5 |
| `qwen/` | Qwen2、Qwen3 及现有变体 |
| `stepfun/` | Step 3.5、3.7 |
| `tencent/` | HY3 |
| `zhipu/` | GLM 5.2 |

例如，`deepseek/deepseek_v4_w4a8` 对应
`configs/deepseek/deepseek_v4_w4a8.json` 与 `rules/deepseek/deepseek_v4_w4a8.json`。
文件名统一为小写下划线形式的 `<模型名>_<量化方案>.json`，版本小数点也使用下划线，
例如 `minimax_m2_5_w4a8.json`。可选模型变体位于方案前，FP8、W4A8、W8A8 配方分别保留。
量化参数保持不变。
厂商分组用于检索；蒸馏或衍生模型应按实际结构与权重格式选择配方。
配方存在不表示所有该系列模型或硬件架构都已完成精度验证。

## 使用配方

先通过只读匹配报告核对目标张量，再使用独立输出目录转换：

```bash
python examples/iluvatar-compressed-tensors/quant.py \
  --input /path/to/original-checkpoint \
  --recipe deepseek/deepseek_v4_w4a8 \
  --match-only --match-report /path/to/rule-match.json

python examples/iluvatar-compressed-tensors/quant.py \
  --input /path/to/original-checkpoint \
  --output /path/to/quantized-checkpoint \
  --recipe deepseek/deepseek_v4_w4a8 \
  --device cuda
```

`python examples/iluvatar-compressed-tensors/quant.py --help` 列出可选配方。
添加配方时，在 `configs/<vendor>/` 与 `rules/<vendor>/` 中加入同名 JSON，
无需再维护名称注册表。本次目录调整不改变张量匹配、反量化、量化或打包逻辑。

## 显式路径与兼容性

原有 `--configs` 与 `--rules` 仍可成对使用，也支持仓库外的自定义 JSON：

```bash
python examples/iluvatar-compressed-tensors/quant.py \
  --input /path/to/original-checkpoint \
  --output /path/to/quantized-checkpoint \
  --configs examples/iluvatar-compressed-tensors/configs/deepseek/deepseek_v4_w4a8.json \
  --rules examples/iluvatar-compressed-tensors/rules/deepseek/deepseek_v4_w4a8.json \
  --device cuda
```

CLI 保留所有历史内置名称，包括旧平铺路径、旧厂商目录路径和旧 `--recipe` 名称。
例如以下用法均选择同一套配方：

```text
--recipe deepseek/deepseek_v4_w4a8
--recipe deepseek/deepseek_v4
--configs .../configs/deepseek/deepseek_v4_w4a8.json --rules .../rules/deepseek/deepseek_v4_w4a8.json
--configs .../configs/deepseek/deepseek_v4.json --rules .../rules/deepseek/deepseek_v4.json
--configs .../configs/deepseek_v4.json --rules .../rules/deepseek_v4.json
```

其中 `...` 代表本工具所在目录，不是字面命令参数；相对路径仍相对于调用者的工作目录。
实际存在的文件优先，自定义路径不会被重定向；若平铺名称对应多套配方，需写明厂商路径。
`--help` 仅展示规范名称。旧名称映射由 `recipes.py` 的 `LEGACY_RECIPES` 统一维护，
重命名时保留旧条目并直接指向最终名称，不建立多级别名。
直接调用 Python 的 `convert()` / `match_only()` 时，应传入实际存在的新路径。
`--recipe` 不与 `--configs` / `--rules` 混用，避免隐式覆盖配方的一半。

## 量化方案后缀

- `w4a8`：MoE 配方中指选中的 routed experts 使用 W4A8，普通选中线性层使用 W8A8；
  dense 配方则指选中的普通线性层使用 W4A8。未选中张量仍按 JSON 规则处理。
- `w8a8`：选中的量化线性层使用 W8A8；权重排布等以 JSON 为准。
- `fp8`：使用 FP8 配方，不表示整个模型的所有张量都会重新量化。
- `fp8_w8a8`：专家保留源 FP8，选中的普通线性层转换为 INT8 W8A8。

MiniMax M2.5 的 FP8 配方保留源专家，只将选中 attention 权重从 block FP8 转为
per-channel FP8；Step 3.7 Flash 的 FP8 配方将选中 BF16 线性层转为 per-channel FP8。
`step3_7_flash_fp8_w8a8` 与 `step3_7_flash_mix_fp8_w8a8` 保留为两套独立配方：
前者普通 MLP 运行时 target 为 `re:model\.layers\.[0-2]...`，后者为 `re:.*\.[0-2]...`。
本次不合并或修改这些匹配规则。

## 历史名称对照

下表名称不带 `.json`；相同映射同时适用于 configs、rules 与 `--recipe`。
未列出的名称保持不变。

| 历史名称 | 规范名称 |
| --- | --- |
| `deepseek/deepseek_r1` | `deepseek/deepseek_r1_w4a8` |
| `deepseek/deepseek_v32` | `deepseek/deepseek_v3_2_w4a8` |
| `deepseek/deepseek_v4` | `deepseek/deepseek_v4_w4a8` |
| `minimax/minimax_m25` | `minimax/minimax_m2_5_w4a8` |
| `minimax/minimax_m25_fp8` | `minimax/minimax_m2_5_fp8` |
| `minimax/minimax_m27` | `minimax/minimax_m2_7_w4a8` |
| `minimax/minimax_m3` | `minimax/minimax_m3_w4a8` |
| `moonshot/kimi_k25` | `moonshot/kimi_k2_5_w4a8` |
| `qwen/qwen36-35b-a3b` | `qwen/qwen3_6_35b_a3b_w4a8` |
| `qwen/qwen3_5` | `qwen/qwen3_5_w4a8` |
| `stepfun/step3p5_w4a8` | `stepfun/step3_5_w4a8` |
| `stepfun/step3p7_flash_fp8` | `stepfun/step3_7_flash_fp8` |
| `stepfun/step3p7_flash_w8a8` | `stepfun/step3_7_flash_fp8_w8a8` |
| `stepfun/step3p7_flash_w8a8_fp8_mix` | `stepfun/step3_7_flash_mix_fp8_w8a8` |
| `stepfun/step3p7_w4a8` | `stepfun/step3_7_w4a8` |
| `tencent/hy3` | `tencent/hy3_w4a8` |
| `zhipu/glm_52` | `zhipu/glm5_2_w4a8` |

## 验证

```bash
python -m unittest discover -s examples/iluvatar-compressed-tensors/tests -v
```

回归测试使用微型 CPU checkpoint，检查新入口、显式路径、旧路径与自定义配方的
匹配和转换结果；不启动模型评测或 GPU 任务。发布权重前仍需单独完成模型精度验证。

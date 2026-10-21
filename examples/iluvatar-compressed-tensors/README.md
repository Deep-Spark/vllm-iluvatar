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

例如，`deepseek/deepseek_v4` 对应
`configs/deepseek/deepseek_v4.json` 与 `rules/deepseek/deepseek_v4.json`。
现有配方名称及量化参数保持不变；FP8、W4A8、W8A8 等不同配方仍各自独立。
厂商分组用于检索；蒸馏或衍生模型应按实际结构与权重格式选择配方。
配方存在不表示所有该系列模型或硬件架构都已完成精度验证。

## 使用配方

先通过只读匹配报告核对目标张量，再使用独立输出目录转换：

```bash
python examples/iluvatar-compressed-tensors/quant.py \
  --input /path/to/original-checkpoint \
  --recipe deepseek/deepseek_v4 \
  --match-only --match-report /path/to/rule-match.json

python examples/iluvatar-compressed-tensors/quant.py \
  --input /path/to/original-checkpoint \
  --output /path/to/quantized-checkpoint \
  --recipe deepseek/deepseek_v4 \
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
  --configs examples/iluvatar-compressed-tensors/configs/deepseek/deepseek_v4.json \
  --rules examples/iluvatar-compressed-tensors/rules/deepseek/deepseek_v4.json \
  --device cuda
```

CLI 也兼容迁移前的内置平铺路径，例如 `configs/deepseek_v4.json`：
当旧路径不存在时，按相同文件名查找厂商目录；相对路径仍相对于调用者的工作目录。
已存在的文件和自定义路径不会被重定向；若同名配方存在多个厂商目录，需写明新路径。
直接调用 Python 的 `convert()` / `match_only()` 时，应传入实际存在的新路径。
`--recipe` 不与 `--configs` / `--rules` 混用，避免隐式覆盖配方的一半。

## 验证

```bash
python -m unittest discover -s examples/iluvatar-compressed-tensors/tests -v
```

回归测试使用微型 CPU checkpoint，检查新入口、显式路径、旧路径与自定义配方的
匹配和转换结果；不启动模型评测或 GPU 任务。发布权重前仍需单独完成模型精度验证。

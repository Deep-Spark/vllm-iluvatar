# vllm-iluvatar

> Iluvatar GPU 上的 vLLM Out-of-Tree 插件

[English](README.md) | [简体中文](README.zh.md)

## 项目简介

`vllm-iluvatar` 是基于 vLLM 官方
[插件系统](https://docs.vllm.ai/en/latest/design/plugin_system.html)
开发的硬件适配插件，通过 `vllm.general_plugins` entry_point 在 vLLM 启动时
自动加载，无需修改 vLLM 主仓源码，就能在 Iluvatar GPU 上跑通
**DeepSeek-V4-Flash W4A8 (v1-TN)** 推理。

**当前 Scope**：本插件仅承载 `DeepSeek-V4-Flash w4a8 (v1-TN)` 一个变体所必需的
增量改动。其他 Iluvatar 已支持的模型/特性仍在 vLLM 的 `iluvatar_main` fork 中维护。

## 更新历史

- **[2026-05] v0.1.0** —— 首发，支持 `IluDeepseekV4ForCausalLM` 模型架构与
  `ilu_deepseek_v4_w4a8` 量化方法；通过 6 处 `sys.modules` 重定向 + 4 处属性级
  monkey-patch 完成 vLLM 上游到 DeepSeek-V4 W4A8 路径的桥接。

## 前置条件

- **硬件**：Iluvatar GPU
- **操作系统**：Linux
- **软件**：
  - Python `>= 3.10`
  - vLLM：必须切到 **`iluvatar_main` 分支**
  - PyTorch / ixformer / ixinfer：版本与 vLLM `iluvatar_main` 分支保持一致

## 支持模型

| 模型 | 量化 | 架构名 / 备注 |
| --- | --- | --- |
| DeepSeek-V4-Flash | W4A8 (v1-TN) | `IluDeepseekV4ForCausalLM` + `ilu_deepseek_v4_w4a8` |

## 安装

下面 `{{VLLM_REPO}}` 表示你本地 vLLM 仓库路径，`{{PLUGIN_REPO}}` 表示本仓库路径。

```bash
# 1. 确认 vLLM 在 iluvatar_main 分支（未打 dsv4 补丁）
cd {{VLLM_REPO}}
git checkout iluvatar_main

# 2. 以可编辑模式安装插件
cd {{PLUGIN_REPO}}
pip install -e .

# 3. 自检（无需 checkpoint，CPU 环境也可跑）
python scripts/smoke_test.py
```

`scripts/smoke_test.py` 会做 6 项断言：entry_point 可发现、模型架构已注册、
量化方法已注册、6 个 `sys.modules` 重定向生效、`vllm._custom_ops` patch 生效、
`ModelConfig.__post_init__` 已包装。**全部输出 `[OK  ]`** 才算装好。

## 开发

clone 之后装一次 pre-commit，所有 commit 自动跑 black / isort + 文本卫生
检查（行尾空白、LF 换行、大文件守门等）：

```bash
pip install pre-commit
pre-commit install
```

按需对全仓跑一遍：

```bash
pre-commit run --all-files
```

## 快速开始

> 默认参数与 `llm-modelzoo` 仓库下
> `inference/vllm/models/DeepSeek-V4/DeepSeek-V4-Flash-w4a8-v1-TN/config.sh`
> 一致。

### 在线推理

```bash
vllm serve {{MODEL_PATH}}/DeepSeek-V4-Flash-w4a8-v1-TN \
  --tensor-parallel-size 8 \
  --trust-remote-code \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 2048 \
  --port 12345
```

健康检查：

```bash
curl http://127.0.0.1:12345/v1/models
```

简单请求：

```bash
curl http://127.0.0.1:12345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "{{MODEL_PATH}}/DeepSeek-V4-Flash-w4a8-v1-TN",
    "messages": [{"role": "user", "content": "用中文简单介绍一下 deepseek?"}],
    "temperature": 0.0,
    "max_tokens": 128
  }'
```

### 一键脚本

`llm-modelzoo` 仓库已封装好「起服务 → 探活 → 跑评测/压测 → 退出清理」的脚本：

```bash
cd {{LLM_MODELZOO_REPO}}/inference/vllm/models/DeepSeek-V4/DeepSeek-V4-Flash-w4a8-v1-TN

bash run_mmlu_accuracy.sh           # MMLU 精度评测
bash run_perf_serving.sh            # 性能压测（默认 1 路 2048→1024）
START_SERVER=0 bash run_perf_serving.sh --max-concurrency 64 --num-prompts 256
```

## 验证插件已生效

### 方法 1：抓日志关键字

`vllm serve` 启动日志里应该出现以下 5 行 `vllm-iluvatar:` 开头的 patch 记录：

```text
vllm-iluvatar: installed sys.modules redirects for 6 modules
vllm-iluvatar: patched vllm._custom_ops (silu_and_mul_with_clamp, router_gemm_bf16_fp32, topk_hash_softplus_sqrt, dsa_convert_req_index_to_global_index)
vllm-iluvatar: patched vllm.v1.core.kv_cache_utils (_get_kv_cache_config_deepseek_v4 + get_kv_cache_config_from_groups)
vllm-iluvatar: defaulting tokenizer_mode='deepseek_v4' for IluDeepseekV4ForCausalLM
vllm-iluvatar: registered model architecture IluDeepseekV4ForCausalLM
```

一行都没有 ⇒ 插件没装上或没被加载，先看下面《常见问题》。

### 方法 2：运行时探针

```python
import vllm.plugins as p
p.load_general_plugins()

from vllm import ModelRegistry
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

assert "IluDeepseekV4ForCausalLM" in ModelRegistry.get_supported_archs()
assert "ilu_deepseek_v4_w4a8" in QUANTIZATION_METHODS
```

## 进一步阅读

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) —— 插件加载顺序、`patch/` 与
  `_replacements/` 的二分设计、6 个整文件替换模块的清单与原因、升级维护指南
- [vLLM 插件系统官方文档](https://docs.vllm.ai/en/latest/design/plugin_system.html)
- [vLLM 模型注册](https://docs.vllm.ai/en/stable/contributing/model/registration/)

## License

Apache-2.0，详见 [LICENSE](LICENSE)。

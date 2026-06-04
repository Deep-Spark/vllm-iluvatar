<h3 align="center">
vLLM Iluvatar Plugin
</h3>

<p align="center">
| <a href="docs/ARCHITECTURE.md"><b>架构说明</b></a> | <a href="https://docs.vllm.ai/en/latest/design/plugin_system.html"><b>vLLM 插件系统</b></a> |
</p>

<p align="center">
<a href="README.md">English</a> | <b>中文</b>
</p>

---

## 最新动态

- [2026-06] v0.2.0 —— 支持50+模型。
- [2026-05] v0.1.0 —— 首发，仅支持 DeepSeek-V4。

---

## 概览

`vllm-iluvatar` 是一个 Out-of-Tree 插件，让 vLLM 跑在天数智芯（Iluvatar）GPU 上。

## 前置条件

- **硬件**：Iluvatar GPU
- **操作系统**：Linux
- **软件**：
  - Python `>= 3.10`
  - vLLM：社区版
  - PyTorch / ixformer / ixinfer：与天数运行时保持一致

## 快速开始

下面 `{{PLUGIN_REPO}}` 表示本仓库路径。

```bash
# 1. 安装社区版 vLLM

# 2. 以可编辑模式安装插件（叠加在社区 vLLM 之上）
cd {{PLUGIN_REPO}}
pip install -e .

# 3.（天数上推荐）安装解释器启动期 bootstrap，使镜像树 finder 在 import vllm 之前生效，
#    这是保留 ixformer 通信优化的前提。
python3 scripts/install_bootstrap.py

```

## 运行模型

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

健康检查与简单请求：

```bash
curl http://127.0.0.1:12345/v1/models

curl http://127.0.0.1:12345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "{{MODEL_PATH}}/DeepSeek-V4-Flash-w4a8-v1-TN",
    "messages": [{"role": "user", "content": "用中文简单介绍一下 DeepSeek?"}],
    "temperature": 0.0,
    "max_tokens": 128
  }'
```

## License

Apache-2.0，详见 [LICENSE](LICENSE)。

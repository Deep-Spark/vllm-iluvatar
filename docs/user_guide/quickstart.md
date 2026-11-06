# 快速开始



示例模型是 Qwen3。本页用它走最短路径：启动一个 OpenAI 兼容服务，确认模型列表，再发一次对话请求。

## 前置条件



已经安装 `vllm-iluvatar`。尚未安装时，打开侧边栏的「安装」，按其中的安装页继续。本页不复制安装命令。

另外确认：

- 已完成[安装自检](installation/installation_check.md)，`vllm`、`vllm-iluvatar` 和插件 entry point 正常。
- 已能访问模型权重目录，例如 `/data/models/Qwen3`。
- 已确认服务端口未被占用。下面示例使用 `12345`。

权重需要自己准备，本页不提供下载。`/data/models/Qwen3` 只是路径示例，模型是否支持见[支持模型](supported_models.md)。

## 启动 Qwen3



打开第一个 shell，用最小参数启动 vLLM 服务。以下模型路径只是示例，请换成实际目录。

```bash
export MODEL_PATH=/data/models/<your-model-dir>
export PORT=12345

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --served-model-name test
```

启动后保持该 shell 不退出，等待日志显示服务监听完成。

下面两次检查用的地址、端口和模型 id 都来自本次启动命令。本页示例是 `http://127.0.0.1:12345`，模型 id 是 `test`。DeepSeek 等其他示例更换这三项，端点 `/v1/models` 和 `/v1/chat/completions` 不变。

## 检查模型列表



```bash
curl -s "http://127.0.0.1:12345/v1/models"
```

成功标准：返回的 `id` 与本次 `--served-model-name` 一致。

## 发起一次对话请求



打开第二个 shell，向 OpenAI 兼容接口发起请求：

```bash
curl "http://127.0.0.1:12345/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @- <<'JSON'
{
  "model": "test",
  "messages": [
    {
      "role": "user",
      "content": "请用三句话介绍 vllm-iluvatar。"
    }
  ],
  "max_tokens": 128,
  "temperature": 0,
  "stream": false
}
JSON
```

成功标准：响应里有 `choices`。不要求出现某一句固定回答。

!!! note "为什么用这个例子"
    本页的目标是“第一次跑通”，不是跑大模型或压测，所以这里特意选择一个加载快、单机能跑的 Qwen3 dense 模型，并把参数压到最小：

    - `--tensor-parallel-size 1`：单卡即可启动；多卡时按设备数量调大，一般不需要 `--pipeline-parallel-size`（流水并行用于跨多机或超大模型）。
    - `--enforce-eager`：跳过 CUDA Graph，启动路径最短，最容易定位环境和模型加载问题。
    - 其余参数（上下文长度、调度、kernel 环境变量等）都用默认值，不在本页堆调优项。

    大模型、MoE、量化、投机解码等场景有各自的推荐参数，不适合直接套用本页示例。

## 其他模型



不同模型的推荐启动参数不同，差异主要来自并行方式、量化格式、是否走投机解码（MTP 是投机解码的一种）、是否启用 CUDA Graph，以及 Model Runner V1 / V2 的选择。本页示例不适用于所有模型：

- 各模型支持范围和注意事项见[支持模型](supported_models.md)。
- 启动参数、环境变量和各项能力见[配置与功能](config_features/index.md)。
- 具体模型的完整启动示例（如 DeepSeek V4）、精度验证和性能压测见[示例](examples/index.md)。

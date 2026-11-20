# GLM-5.3-Flash 在 CQ 上的 P/D 启动说明

在 CQ 服务容器内使用本分支的 vLLM 和插件。Prefill（P）和 Decode（D）默认直接读取原始 E4M3 128 × 128 block-FP8 模型目录（在加载时完成 routed experts FP4 打包及投影量化），无需离线转换；若传入由 `quantize_fp8_to_ilu_f4f8.py` 生成的离线 `ilu_f4f8` 目录，vLLM 也会根据 `config.json` 自动识别。以下命令在仓库的 `examples/glm5next/` 下执行。

## 1. 启动 Prefill

在 P 所用机器的服务容器执行。示例芯片顺序取自 18.10 的前八芯，换机时按本机拓扑填写 `DEVICES`。两条启动命令均在前台运行，P 和 D 各使用一组八芯。

```bash
MODEL_PATH=/models/GLM-5.3-Flash \
DEVICES=0,1,2,3,6,7,4,5 HOST=0.0.0.0 PORT=8000 \
bash run_cq_prefill.sh --serve
```

## 2. 启动 Decode

在 D 所用机器的服务容器执行，`MODEL_PATH` 指向该容器可见的原始 FP8 模型目录。

```bash
MODEL_PATH=/models/GLM-5.3-Flash \
DEVICES=0,1,2,3,6,7,4,5 HOST=0.0.0.0 PORT=8001 \
bash run_cq_decode.sh --serve
```

## 3. 可选：从原始 FP8 权重生成离线 `ilu_f4f8` 模型

如需提前离线打包权重以缩短服务冷启动时间，可执行：

```bash
python3 quantize_fp8_to_ilu_f4f8.py \
  --input /models/GLM-5.3-Flash \
  --output /models/GLM-5.3-Flash-ilu_f4f8
```

两个入口的服务模型名均为 `GLM-5-Next`。将 `--serve` 换成 `--print-command` 可查看 vLLM 参数。复现下表配置时，使用干净的服务 shell，由脚本设置模型参数。

| 配置 | Prefill | Decode |
|---|---|---|
| 并行 | TP1 / PP8 / DP1，启用 EP，流水线分层 `7,6,6,6,6,5,5,4` | TP2 / PP1 / DP4，启用 EP |
| 执行 | eager，异步调度，chunked prefill | `FULL_DECODE_ONLY` Graph，同步调度 |
| MoE 通信 | AllGather/ReduceScatter | AllGather/ReduceScatter |
| MTP | 1 个 draft token，复用 index | 3 个 draft tokens，复用 index |
| KV / KDA | vLLM KV 配置 BF16，indexer KV 和 KDA state 为 BF16，关闭 replaySSM | 同样的 dtype，开启 replaySSM |
| 最大上下文 / 每批 token 预算 / 最大序列数 | 131,072 / 2,304 / 64 | 278,528 / 2,048 / 32 |
| 显存利用率参数 | 0.90 | 0.90 |

## 4. 已有性能

2026-11-19 在 CQ 18.10 使用上述离线权重和配置，各运行一轮腾讯微观脚本。P 和 D 各用八芯（四张双芯物理卡），核心频率 1.8 GHz、显存频率 1.6 GHz，ACS 关闭，GTS 已开启。

| 场景 | 负载 | 该轮实测 | 服务日志中的 KV 容量 |
|---|---|---|---|
| P | CC8，每请求实际未命中缓存的输入为 8,064 tokens | **26,041.78 未缓存输入 tok/s**，TTFT P50 为 1.792 s，P90 为 2.377 s | 单个 PP8 引擎 23,074,021 tokens |
| D | CC64，每个 DP 16 个请求，输入 112,640 tokens，目标请求的测量窗口由背景请求全程覆盖，抢占数为 0 | **45.30 ms/step** | 每个 DP 4,233,145 tokens，四个 DP 合计 16,932,580 tokens |

D 按每步平均产出 **3.2 tokens** 换算，TPOT 为 **14.16 ms**（45.30 / 3.2），CC64 总吞吐为 **4,520.97 tok/s**（64 × 3.2 × 1000 / 45.30），每张双芯物理卡 **1,130.24 tok/s**。3.2 是换算假设。

该次实测使用 vLLM 0.31.0 `341bf7d4d124`，vllm-iluvatar `e8213aead732`，IxFormer `0def4d4fc832`，ixinfer `d1cfe6556d28`，ixmm `df9e13eaf4f7`，ixattn `2a9720747cfb`。在其他 CQ 机器复现时使用与宿主驱动匹配的 CoreX 环境。18.10 当次 D 服务使用了提供 Graph capture 接口兼容的 `libcaptureinfo_compat.so`，由该机环境中的 vLLM 入口加载。

# vLLM bench serve 压测



本示例展示如何使用 `vllm bench serve` 对 OpenAI 兼容服务做随机数据压测。模型路径、并发数、输入输出长度和结果目录需要按实际目标调整。

## 随机输入输出压测



```bash
vllm bench serve \
  --base-url "http://127.0.0.1:12345" \
  --model /data/DeepSeek-V4 \
  --served-model-name test \
  --backend openai \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --metric-percentiles 50,90,99 \
  --max-concurrency 4 \
  --num-prompts 40 \
  --save-result \
  --result-dir ./bench_results \
  --result-filename isl1024_osl1024_c4.json \
  --ignore-eos
```

## Decode Only 场景



Decode Only 场景可使用 prefix cache：

```bash
vllm bench serve \
  --base-url "http://127.0.0.1:12345" \
  --model /data/DeepSeek-V4 \
  --served-model-name test \
  --backend openai \
  --dataset-name random \
  --random-input-len 64 \
  --random-prefix-len 8192 \
  --random-output-len 1024 \
  --metric-percentiles 50,90,99 \
  --max-concurrency 4 \
  --num-prompts 40 \
  --num-warmups 1 \
  --save-result \
  --result-dir ./bench_results \
  --result-filename prefix8192_osl1024_c4.json \
  --ignore-eos
```


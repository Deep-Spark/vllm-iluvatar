# EvalScope 精度验证



本示例展示如何通过 OpenAI API 模式使用 EvalScope 做精度验证。数据集路径、批大小、超时时间和生成参数需要按实际验证目标调整。

```bash
evalscope eval \
  --model test \
  --api-url "http://127.0.0.1:12345/v1" \
  --api-key EMPTY \
  --eval-type openai_api \
  --eval-batch-size 128 \
  --datasets math_500 \
  --dataset-dir ./datasets \
  --generation-config '{"timeout":900,"max_completion_tokens":8192,"max_tokens":8192,"temperature":0}' \
  --work-dir ./results_math_500 \
  --no-timestamp
```


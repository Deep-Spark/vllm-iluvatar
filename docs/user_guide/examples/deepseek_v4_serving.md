# DeepSeek V4 服务启动



以下命令是举例说明，环境变量和模型路径按实际输入。

## 启动命令



```bash
MODEL=/data/DeepSeek-V4

export VLLM_ILU_DS4_PREFILL_AS_DECODE=1

vllm serve "${MODEL}" \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --data-parallel-size 1 \
  --port 12345 \
  --gpu-memory-utilization 0.9 \
  --no-enable-prefix-caching \
  --max-model-len 16384 \
  --max-num-batched-tokens 4096 \
  --enforce-eager \
  --served-model-name test
```

需要确认服务时，按[快速开始](../quickstart.md)的验证步骤检查，并把地址、端口和模型 id 换成本次启动命令里的值。

## 后台运行



如需后台运行并保存日志，不用另换一套参数，在上面的启动命令末尾加上重定向即可（给原来的最后一行补上续行符 `\`）：

```bash
  --served-model-name test \
  > vllm-ds4.log 2>&1 &
```


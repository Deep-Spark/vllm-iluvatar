# Model Runner V2



`vllm-iluvatar` v0.27.1 中 Model Runner V2 已可用。部分功能组合会影响实际运行路径。

## 默认行为



- v0.27.1 起使用 Model Runner V2。
- 如需显式使用 V2，可设置 `VLLM_USE_V2_MODEL_RUNNER=1`。

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

- 如需强制回到 V1，可设置 `VLLM_USE_V2_MODEL_RUNNER=0`。

```bash
export VLLM_USE_V2_MODEL_RUNNER=0
```

## 量化 KV cache



量化 KV cache 不支持 V2，需显式设置 `VLLM_USE_V2_MODEL_RUNNER=0`。

## 使用建议



在量化 KV cache 等场景中，Model Runner 路径可能不同，可结合实际模型和部署目标选择。

使用 Model Runner V2 时开启 PP（流水线并行）可能导致性能下降；如果模型需要开 PP，建议使用 Model Runner V1。


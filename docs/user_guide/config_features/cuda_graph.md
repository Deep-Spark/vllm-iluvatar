# CUDA Graph



`vllm-iluvatar` v0.27.1 默认开启 CUDA Graph，当前只支持 `mode=0` 与 `FULL_DECODE_ONLY` 这一种组合。

## 推荐配置



如果启用 CUDA Graph，启动时使用如下编译配置：

```bash
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","mode":0}'
```

## 排查建议



- 快速验证或定位环境问题时，可以先使用 `--enforce-eager` 排除 CUDA Graph 影响。
- 量化模型、DP / EP、多卡大模型场景下，如果出现显存、启动或 stream capture 相关问题，优先确认 CUDA Graph 配置是否符合当前版本要求。
- 不要继续使用旧的 `ENFORCE_CUDA_GRAPH` 环境变量；相关行为已改由启动参数控制。


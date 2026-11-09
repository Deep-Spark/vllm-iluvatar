# 量化



`vllm-iluvatar` v0.27.1 对量化格式和配置方式做了升级。使用 W4A8 / W8A8 权重前，需要确认权重和配置是否符合本版本要求。

## 量化格式



v0.27.1 的权重量化走 Compressed Tensors：`quant_method` 是 `compressed-tensors`，`format` 是 `iluvatar-mixed`。`pack_version=2` 只出现在写出了该字段的组上，不是每个 W4A8 / W8A8 配置都有。

旧版本 W4A8 / W8A8 权重不能直接沿用，需要使用新版量化脚本和配置重新做离线量化，详见下一节。

## 旧权重重新量化



如果已有旧版 W4A8 / W8A8 checkpoint，需要重新执行离线量化流程。离线量化的入口为：

- [iluvatar-compressed-tensors/quant.py](../../../examples/iluvatar-compressed-tensors/quant.py)
- 新版 [configs](../../../examples/iluvatar-compressed-tensors/configs)
- 新版 [rules](../../../examples/iluvatar-compressed-tensors/rules)

本版本提供覆盖 16 个模型族的 CT rules / configs：DeepSeek R1 / V3.1、DeepSeek V3.2、DeepSeek V4、GLM-5.2、Kimi-K2.5、MiniMax-M2.5 / M2.7 / M3、Qwen2、Qwen3、Qwen3.5、Qwen3.6-35B-A3B、Qwen3.6-MoE、Qwen3.8、Step-3.5 / 3.7。具体模型和配置文件以随版本发布的内容为准。

## CT config / rule 是什么



CT 即 Compressed Tensors。config 解析后进入 checkpoint 的 `quantization_config`；rule 只参与离线量化，不写入 checkpoint。量化脚本读取成对的 config 与 rule，把结果写入新的 W4A8 / W8A8 checkpoint；W4A8 / W8A8 的方法和布局由 config 里的 `config_groups.<group>.iluvatar` 块描述，不再通过旧环境变量控制。量化产物写入 checkpoint 的 `config.json`，`vllm-iluvatar` 插件在加载时自动读取，正常使用无需手工干预这些字段。

`iluvatar` 块的字段语义（`gemm_format`、`pack_version`、`method` 等）与自定义量化格式的扩展方式见[开发者文档 / 扩展量化](../../developer_guide/extend_quantization.md)。

## 启动环境变量清理



v0.27.1 删除了一批旧环境变量，相关行为改由 CT config 的 `iluvatar` 块或启动参数控制。

已删除的环境变量包括：

- `VLLM_MOE_W8A8_W4A8`，原 `VLLM_W8A8_MOE_USE_W4A8`
- `VLLM_W4A8_FORMAT`
- `VLLM_W4A8_VERSION`
- `VLLM_W8A8_FORMAT`
- `VLLM_W8A8_GEMM_MODE`
- `ENFORCE_CUDA_GRAPH`

因此，启动量化模型时不需要再额外传入这些旧量化环境变量。



权重量化看 CT config。KV cache 量化仍由 `VLLM_ILU_FLASH_ATTN_*` / `VLLM_ILU_MLA_ATTN_QUANT_TYPE` 控制，且要搭配 `VLLM_USE_V2_MODEL_RUNNER=0`（见[启动参数与环境变量](launch_args_and_env.md)）。

## 运行时量化 `ilu_runtime`



本版支持对未量化 checkpoint 做在线量化：以 `--quantization ilu_runtime` 指定未量化 checkpoint 作为挂载点，由 `additional-config.ilu_quant` 提供转换配置，插件在加载时完成转换。已带量化配置的模型不要加 `--quantization ilu_runtime`。

需要注意：未量化 checkpoint 若忘记 `--quantization ilu_runtime`，服务可以正常 READY 但全程以 BF16 运行、不报错。启用前先确认模型、精度和误差已经验证。完整命令、支持模型与限制以随版本发布的运行时量化指南为准。

## 注意事项



- 旧 W4A8 / W8A8 checkpoint 必须重新量化后再用于 v0.27.1。
- 不要继续依赖已删除的旧环境变量。
- 优先让量化信息随 checkpoint 的 `config.json` 一起被插件读取，减少启动命令中的手工配置。
- 使用 `ilu_runtime` 前，需要先确认模型、精度、性能和精度误差是否已经验证。

## 第三方量化格式（awq / GPTQ）



除天数自研的 `iluvatar-mixed`（W4A8 / W8A8，CT config）外，本版还支持直接加载社区常见的第三方量化权重：awq 和 GPTQ。这是与 `iluvatar-mixed` 不同的另一条路线。

- **awq**：走默认加载路径即可，例如 Qwen3.5-35B-A3B-AWQ。
- **GPTQ**：需要设置线性层后端，例如 Qwen3-235B-A22B-GPTQ-Int4：

```bash
export VLLM_ILU_LINEAR_BACKEND=gptq_marlin
vllm serve /path/to/Qwen3-235B-A22B-GPTQ-Int4 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --trust-remote-code
```

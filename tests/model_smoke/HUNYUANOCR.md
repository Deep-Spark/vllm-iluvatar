# HunyuanOCR 单图验证

模型通过上游 Transformers 多模态后端加载，架构为
`HunYuanVLForConditionalGeneration`。本入口独立运行，不加入常规 CI 的
`suite.json`。完整 OCR 支持需要真实权重的 20/20 验收；随机权重执行通过
只说明代码路径可运行。

当前验证目标为官方 `tencent/HunyuanOCR` 根目录的 HunyuanOCR 1.5，固定
revision `47644ecc4fc854efa4f505155158831f36773ee4`，不是其 `v1.0/`
子目录。官方权重由 ModelHub 导入，开发验证从固定发布快照只读加载。
2026-11-24 已使用真实权重完成 BF16 Eager 验收：20/20 请求正常结束，
20/20 答案完整匹配。使用 ModelHub 固定快照
`/models/.snapshots/model/tencent--HunyuanOCR/v1`，平台 revision
`4bd7dca8c4246c4f2c29fdbc2e9945067cfeb281bd68b66be9ff01dd284b1a80`。

适配层默认设置 `use_fast=False`，采用官方示例使用的 PIL 图像处理器。
Transformers 5.14.1 的 fast 处理器按 2×2 merge block 排列 patch，
而此模型的 PIL 路径按行排列；形状一致但内容顺序错误，会导致 OCR 失真。
实测把 fast 输出重排为按行顺序后，与 PIL 输入逐元素相等。
显式传入的 processor 选项仍被保留，便于验证上游修复；当前验收不包含
`use_fast=True`。

在符合项目设备占用检查要求的容器中，只使用分配的逻辑 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/ixmm/lib:${LD_LIBRARY_PATH:-}"

# MODEL_PATH 指向 ModelHub 的完整固定快照，不使用带残缺下载文件的目录。
vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 --port 18123 --served-model-name hunyuanocr-test \
  --dtype bfloat16 --max-model-len 4096 --max-num-seqs 4 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.5 \
  --limit-mm-per-prompt '{"image":1}' --enforce-eager
```

上述 Eager 配置已通过真实权重的单图 20/20 验收。`FULL_DECODE_ONLY` 在当前
Transformers 5.14.1 / CoreX 5.1 环境图捕获时失败：动态 RoPE 的
`dynamic_frequency_update` 对 GPU `seq_len` 执行 Python 条件判断，触发
`operation not permitted when stream is capturing`。当前使用上述 Eager
配置；以下 Graph 命令仅作为问题复现入口，不能作为已支持配置。

复现 Graph 问题时，停止上述服务，在相同权重及其他参数下将
`--enforce-eager` 替换为：

```text
--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
```

两种模式分别保存结果，不能从单一模式通过推断另一模式通过。

客户端需要 Pillow 和支持中文的字体；可使用 `fonts-noto-cjk` 提供的
`NotoSansCJK-Regular.ttc`，或用 `--font` 显式指定字体文件。

```bash
python3 tests/model_smoke/hunyuanocr_smoke.py \
  --base-url http://127.0.0.1:18123 \
  --model hunyuanocr-test --output /tmp/hunyuanocr-eager
```

脚本生成十张中文、英文、数字和混排文档，每个请求一张图。先串行十次，
再以并发四执行同一组十次，无重试。答案完整比较，仅忽略空白；模型名、
单一非空正文、`finish_reason=stop` 都必须正确。生成图片、原始响应和
`result.json` 一起保存，`ocr_passed=true` 才是 20/20 通过。
这只是 OCR 冒烟，不代表表格、公式、多页文档、全量精度或性能验收。

如果只具备官方配置和 tokenizer，可用 `--load-format dummy` 启动随机
权重，并加 `--execution-only` 运行一条八 token 的单图请求。这一模式永远
写出 `ocr_passed=false`，不得将其结果作为真实 OCR 验收证据。

## 本次验收记录

- 机器：[redacted:private-ip]，单个 BI-V150 逻辑 GPU，驱动 CoreX 5.1.0。
- 修复代码：`2c934e646dbe06b5b224c1de40a8ac486ee70e92`；
  main 基线：`88bdb53cd47031baf4e6b47157a68d16abcc56e6`。
- 运行环境：vLLM `0.31.0+corex.5.1.0.20261122`、
  ixformer `0.7.0+corex.5.1.0.20261122`、Transformers `5.14.1`。
  测试镜像 ID：
  `sha256:28c250229ac01f53abd4fb326461b78d81903dab1377c58e1d4c6737c86e78f7`。
- 同一组图片、提示词和判分规则：修复前 fast 路径 18/20 正常结束、
  0/20 正确；修复后默认 PIL 路径 20/20 正常结束、20/20 正确。
  复验没有额外指定 processor，没有诊断插桩、请求重试或跳过多模态预热。
- 85 项 RMSNorm/worker 回归已在此前通过；本次新增预处理及平台注册检查
  21 项通过。服务测试后 health=200，正常退出，GPU 已释放。
- 原始响应、失败基线、环境和权重校验保留在 49：
  `/home/zejun.cai/hunyuanocr-verify-20261124/`。
  本次 49 驱动还依赖匹配的 `libimex_client.so`、`libbmemlink.so`；
  容器运行时已补齐并记录哈希。该结果不等于 CoreX 5.0.1 发布镜像验收。

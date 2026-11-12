# MultiConnector 完成通知来源与重复计数修复

## 原问题

MultiConnector 把同一份完成集合依次交给 LMCache 和 OptNixl 等子连接器。OptNixl 会根据自身的待完成请求表过滤通知；当一个请求仅等待 LMCache 保存完成时，它的合法通知可能被 OptNixl 当作不再属于自己的请求删除。父调度器因此收不到最终完成，API 已返回的直接 P 请求仍保留 GPU KV 块。

另一个风险是同一子连接器跨轮询重复上报，或同一 worker 在其他 worker 尚未完成时重复上报。原计数不能仅凭重复通知次数判断全部参与方已经完成，否则可能过早结束屏障或触发调度器断言。

## 修改内容

改动限定在 vllm-iluvatar 插件，未修改上游 vLLM、LMCache 或定版服务脚本。

1. worker 合并完成集合前，记录每条通知来自哪个子连接器，随原 worker metadata 传给调度器。调度器跨轮询保存来源，直到原有多 worker 完成条件满足。
2. 每个子连接器只处理自己的完成集合，使用浅拷贝的输出对象隔离修改，再合并处理结果。保留 OptNixl 对真实早到完成的阻止释放逻辑、过期通知过滤，以及接收失败和无效块信息。
3. 在尚未结束的发送屏障内，对同一子连接器和固定 worker 槽位的重复通知去重。仍由上游 extra-save 和 rank 计数决定完成；不把完成条件改成当前列表长度，不删除父调度器断言。
4. 插件检查所依赖上游方法的源码指纹。当前适用上游 vLLM 为 `6919d781a67e861e62c6bea2efedd6bdafa410f2`。版本不匹配或发现部分旧补丁时直接报错，避免在未知实现上静默应用。

改动文件：`vllm_iluvatar/distributed/kv_transfer/kv_connector/v1/multi_completion.py`、`vllm_iluvatar/ilu_patch/general_patches/patch_multi_completion_sources.py`。对应 CPU 用例位于 `tests/distributed/test_multi_completion_sources.py` 和 `test_multi_completion_dedup.py`。

## 验证结果

2026-11-13，在 192.168.77.11 的本次任务容器、单机 2P1D 环境验证。基础镜像 ID 为 `99c1f8f795705793d7e8e468f5873cf8914d4b5687de35e2c25d1a3a1e050bed`；vllm-iluvatar 父提交 `051d012c8e`，LMCache `ad569120b0`，lmcache-iluvatar `e94676e670`，服务脚本 `faea90419f`。候选 wheel SHA256 为 `6b511ce148db270f8217963969d693bcc34e3d09b7d25b3cb64250d08b78f95d`，两份产品 Python 文件与实际 GPU 环境加载文件逐项一致，native 文件未改变。

| 验证项 | 结果 |
|---|---|
| CPU 实际方法与计数契约 | 27/27 通过；包括来源隔离、重复计数、早到通知与原屏障行为 |
| 修改前同目标 | 2048-token 直接 P 请求 API 成功，但 P2 的 KV usage 原始指标 `0.001734590181` 在约 63、86、183 秒采样仍未归零 |
| 修改后短请求 | 固定 8 条 HTTP 请求全部成功；直接 P 保存、真实 P/D 冷请求、JSON/SSE 热读通过；热读输出与本代冷请求一致；结束后 0/60/180 秒所选资源归零 |
| 131072-token 长输入 | 冷请求、直接 P 保存、真实 MP 热读成功；热读实际复用 130048 token，889 个键均从 L1 命中，完整输出与冷请求一致 |
| 132096-token 多次保存 | 冷请求、增量保存、热读成功；7 个实际保存 rank 各完成 `32×4096+1024` token 保存；热读复用 131072 token，896 个键均从 L1 命中；末尾资源回到基线 |
| 进程与错误日志 | 上述共 18 条 HTTP 目标请求中，相关进程/context 身份稳定，观察窗内未出现被检查的 8 类严重错误 |

这些结果支持本补丁修复已验证场景中的完成通知丢失和 KV 块不释放。它们不是性能提升比例，也不代表所有稳定性用例已经通过。GPU 重复 rank 故障注入、全部 Decode rank、真实 MP 死亡、健康降级后的物理所有权回收没有在上述目标中验证。

取消请求后空闲 P 不再消费通知的问题使用单独候选 C 验证；本提交不包含 C，也不包含独立心跳候选 H。不能把取消、心跳或所有资源泄露问题一并标为已解决。

## 证据与后续维护

本轮固定请求及逐阶段结果位于 11 宿主机 `/data4/lsun/lmcache_night_20261113` 下的 GPU 验证输出；源文件与 wheel 对照索引保存在共享目录 `/mnt/share/lsun/lmcache_night_20261113/artifacts/fix11_stages/M2/`。完整交付索引 `M2_GPU_TARGET_COMMIT_INPUTS.json` 同时记录 2048、131072/132096 的原件和 CPU 结果，提交时随验证回执保留。

以后 rebase 到新的 vLLM 时，应先复核 MultiConnector、KVOutputAggregator 及 worker 输出槽位契约，再更新源码指纹并重跑 CPU 与 GPU 用例。不能仅修改指纹绕过适配检查。

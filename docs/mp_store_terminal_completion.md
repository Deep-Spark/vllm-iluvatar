# Multi 等待实际的 KV 发送和保存完成

Prefill worker 一边由 OptNixl 把 KV 送给 Decode，一边由 LMCache MP 保存 KV。若 NIXL 的通知先到，当轮元数据还没说明两个子连接器都要等待，Multi 可能提前给父调度器记上这个 worker 的完成。父调度器随后收到其他 worker 的通知，会在这批 KV 保存结束前释放块。

配套组合中，Multi 先记录子连接器实际来源，保留早到的 NIXL 通知。收到 LMCache 终态，并满足本请求所需的不同子连接器数量后才上报一个 worker 完成。父调度器沿用原 rank 数量和释放方法。LMCache Scheduler 以该聚合完成作为延后发送 `END_SESSION` 的条件之一。

## 生效范围

显式设置 `LMCACHE_ILUVATAR_TERMINAL_TRANSFERS=1`，并启用终态 KV 所有权保护。该路径面向 OptNixl 与非 lazy、LMCache-driven 的 LMCacheMP producer，要求多组 KV 按请求 ID 上报读取错误、同步调度和非 speculative 模式。单组使用原有的 block ID 错误处理。配置满足上述条件而缺少配套 LMCache 插件时，新的 Multi 会在初始化阶段报错。部署前需要核对同一个 Prefill 的 Scheduler 和全部 worker 均加载同一组插件。

关闭 LMCache 时，服务直接使用 OptNixl，保留 APC。CPU 测试 `tests/distributed/test_multi_mp_store_barrier.py` 覆盖子连接器先后顺序、重复完成、单组与多组配置及配套插件检查。

## 49 号机验证

当前 vLLM 插件代码在 2P1D 组合定向轮中启用 LMCache。两个 P Scheduler 和 16 个 worker 均实际进入保护路径，冒烟 20/20。实际 Router 请求故意暂停一个 rank 的末次 STORE。该 rank 的 NIXL 先完成，放行 STORE 后 LMCache 报告完成。全部 8 个 P rank 汇合后，该 Prefill 请求只释放一次，随后 MP 确认 END。跨 P 的 JSON 和 SSE 请求各成功一次，首次跨 P 读取外部命中增加 131,072 token。两个定向用例的待回收资源在各检查点为零。

定向轮的同一安装件关闭 LMCache 后，固定 C64 128/128。各请求在 900 秒内返回完整终态，墙钟 1,408.54 秒。配套 LMCache 插件完成期限处理的最后修订后，再关闭 LMCache 验证冒烟 21/21。0/60/180 秒时 P/D 队列为零，MP 端口无监听，并确认同 P 的 APC 复用。主动停机时 D 仍记录已有的同类 SIGSEGV。

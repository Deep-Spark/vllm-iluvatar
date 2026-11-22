# Multi 等待实际的 KV 发送和保存完成

Prefill worker 一边由 OptNixl 把 KV 送给 Decode，一边由 LMCache MP 保存 KV。若 NIXL 的通知先到，当轮元数据还没说明两个子连接器都要等待，Multi 可能提前给父调度器记上这个 worker 的完成。父调度器随后收到其他 worker 的通知，会在这批 KV 保存结束前释放块。

配套组合中，Multi 先记录子连接器实际来源，保留早到的 NIXL 通知。收到 LMCache 终态，并满足本请求所需的不同子连接器数量后才上报一个 worker 完成。父调度器沿用原 rank 数量和释放方法。LMCache Scheduler 以该聚合完成作为延后发送 `END_SESSION` 的条件之一。

## 生效范围

显式设置 `LMCACHE_ILUVATAR_TERMINAL_TRANSFERS=1`，并启用终态 KV 所有权保护。该路径面向 OptNixl 与非 lazy、LMCache-driven 的 LMCacheMP producer，要求多组 KV 按请求 ID 上报读取错误、同步调度和非 speculative 模式。单组使用原有的 block ID 错误处理。配置满足上述条件而缺少配套 LMCache 插件时，新的 Multi 会在初始化阶段报错。部署前需要核对同一个 Prefill 的 Scheduler 和全部 worker 均加载同一组插件。

关闭 LMCache 时，服务直接使用 OptNixl，保留 APC。CPU 测试 `tests/distributed/test_multi_mp_store_barrier.py` 覆盖子连接器先后顺序、重复完成、单组与多组配置及配套插件检查。

## 49 号机验证

当前 vLLM 插件代码在 2P1D 组合定向轮中启用 LMCache。两个 P Scheduler 和 16 个 worker 均实际进入保护路径，冒烟 20/20。实际 Router 请求故意暂停一个 rank 的末次 STORE。该 rank 的 NIXL 先完成，放行 STORE 后 LMCache 报告完成。全部 8 个 P rank 汇合后，该 Prefill 请求只释放一次，随后 MP 确认 END。跨 P 的 JSON 和 SSE 请求各成功一次，首次跨 P 读取外部命中增加 131,072 token。两个定向用例的待回收资源在各检查点为零。

定向轮的同一安装件关闭 LMCache 后，固定 C64 128/128。各请求在 900 秒内返回完整终态，墙钟 1,408.54 秒。配套 LMCache 插件完成期限处理的最后修订后，再关闭 LMCache 验证冒烟 21/21。0/60/180 秒时 P/D 队列为零，MP 端口无监听，并确认同 P 的 APC 复用。主动停机时 D 仍记录已有的同类 SIGSEGV。

## 请求 ID 重用与旧通知

在支持的同步 Prefill 配置中，OptNixl 在产生完成通知时保留本次传输的代际。Multi 通过调度器元数据取得当前有效代际，在 worker 读取完成队列时过滤旧代或缺少代际证明的通知。本代的提前通知仍可参与后续汇合。上一个请求在该 worker 暂存的通知状态在新请求进入时清理。父调度器完成聚合后，worker 根据下一轮调度元数据释放已有完成状态。LMCache 的批次完成路径无需维护按请求 ID 持续增长的历史集合。

同一个内部请求 ID 被重用时，每次 PD 传输必须带不同的传输身份。使用 OptNixl 注册功能的配套路由器会为每次请求生成新的 `transfer_id`。直接构造 PD 参数的客户端必须提供不同的 `transfer_id`。如果省略传输身份而回退到请求 ID，或直接复用前一次身份，两代通知无法区分，属于本功能的适用范围之外。普通无 NIXL 的直接 Prefill 请求不需要该字段。独立 OptNixl 和未启用该开关的路径继续使用原完成通知协议。

若 D 在 P 收到请求前就发起取消，P worker 可能先完成取消处理。开启此功能后，worker 暂存尚未绑定请求的真实代际通知，只在后续 P 接纳相同请求 ID 和传输代际时恢复。即使 P 尚未向 NIXL worker 交付发送任务，已产生的 LMCache 终态也仍可与这条通知一并汇合，供 D 确认释放目标块。旧代或裸通知无法替代当前代。暂存沿用 NIXL 所有权墓碑的容量和过期配置。达到容量时记录错误，保留已有条目，无法保存的通知不会被当成完成。

`tests/distributed/test_multi_generation_fence.py` 用实际 OptNixl 请求代际、发送完成、旧确认重放、队列消费和 Multi 代码，覆盖合法早到、重复、上一代残留、当前代放行、状态退役和关闭功能的兼容性。测试不模拟 GPU 执行。

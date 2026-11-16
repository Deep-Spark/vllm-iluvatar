# 模型冒烟

模型选择、启动配置、问题和判定与产品代码放在同一个PR里。当前覆盖11款固定ModelHub权重；只验证文本启动/作答/小并发，不代表全量精度、多模态或性能验收。已执行真实GPU冒烟；各模型是否通过以对应源码SHA的归档结果为准。

```text
tests/model_smoke/
  suite.json              # 要运行的模型ID及顺序；增删范围改这里
  models/<模型ID>.json    # 权重版本引用、启动参数和测试要求
  weights/<模型ID>.json   # 固定发布快照的文件清单、SHA256和缓存信息
  cases.json              # 10道固定问题与标准答案
  runner.py               # 服务生命周期、请求、判分、投机指标、报告
  test_runner.py          # CPU回归：判分负例、模拟HTTP服务、进程清理
  test_scheduler.py       # CPU回归：并行分配、取消和资源隔离
  README.md
```

新增模型：在weights/保存该ModelHub固定发布版本的文件清单，在models/增加模型JSON，通过weight_manifest引用清单并填写启动参数和测试要求，再把ID加入suite.json，准备对应本地缓存。删除测试范围只需从suite.json去掉ID；不再复用的模型配置和清单可一并删除。修改启动参数只编辑models/下对应JSON；更换权重版本时同时更新版本引用和weights/下的固定快照清单，不需要修改Jenkins或CI仓库的模型名单。每款GPU数量仍限制为TP×PP×DP不超过16。

模型JSON保留name/id、owner/slug/platform_version/revision、weight_manifest，以及logical_gpus/parallel/serve_args/environment和各项测试要求。weight_manifest相对于suite.json所在目录，例如`weights/deepseek-v4-1-flash-1017.json`；只能引用weights/下的普通JSON文件，拒绝越界路径、软链接或与模型配置不一致的权重身份/版本。文件列表、大小、SHA256、缓存路径和READY指纹仅放在weights/，该清单不能覆盖启动参数或测试要求。读取时展开为完整运行记录，选模指纹仍覆盖全部清单内容。

通用判定与服务测试放test_runner.py；以后模型专属的功能/正确性测试可放同目录test_<模型家族>.py，并明确其运行入口。Jenkins调度、GPU锁和权重同步工具的单测属于独立CI仓库，避免把基础设施测试复制到产品库。

## CPU检查与运行入口

```bash
# 不加载权重、不导入vLLM、不使用GPU
python3 tests/model_smoke/runner.py --validate
python3 -m unittest discover -s tests/model_smoke -p 'test_*.py' -v

# 在20检查本地READY版本，不启动模型
python3 tests/model_smoke/runner.py --models qwen3-6-35b-a3b --check-weights

# 已安装本PR构建wheel、并持有CI GPU锁的runtime容器中执行
python3 tests/model_smoke/runner.py --models all --output /work/model-smoke-results
```

普通pytest收集test_runner.py只运行CPU模拟服务测试，不启动真实大模型。真实模型冒烟由CI明确调用runner.py。完整pytest仍遵循产品tests/conftest.py的依赖/设备检查。

## 11 款固定配置

每款只选一个代表配置，逻辑 GPU 上限16。完整命令参数以 models/ 下的模型JSON为准；每次运行将实际命令和环境写入 `launch.json`，除Hy4使用当前支持的eager模式外，其余模型使用 mode=0 的 FULL_DECODE_ONLY。

| model ID | 模型 | TP×PP×DP | 投机解码 |
| --- | --- | --- | --- |
| deepseek-v4-flash | DeepSeek-V4-Flash | 8×1×1 | 无 |
| deepseek-v4-1-flash-1017 | DeepSeek-V4.1-Flash-1017 | 4×4×1 | DSpark 5 |
| deepseek-r1 | DeepSeek-R1 | 4×4×1 | 无 |
| glm-5-3 | GLM-5.3 | 4×4×1 | MTP 3 |
| glm-5-3-flash | GLM-5.3-Flash | 2×8×1 | 无 |
| step-3-5-flash | Step-3.5-Flash | 8×1×1 | 无 |
| minimax-m3 | MiniMax-M3 | 16×1×1 | 无 |
| hy4-preview | Hy4-preview | 8×1×2 | MTP 3 |
| qwen3-6-35b-a3b | Qwen3.6-35B-A3B | 4×1×1 | 无 |
| qwen3-8-27b | Qwen3.8-27B（W8A8） | 4×1×1 | 无 |
| qwen3-8-flash-next | Qwen3.8-Flash-Next | 8×1×1 | 无 |

GLM-5.3 保留发布配置的并行、调度及投机组合；V4.1-1017 按CI要求使用TP4×PP4、DP1，不启用专家并行，使用V2异步调度，保留DSpark 5及原正确率门槛；其余按模型卡缩小上下文/并发/图捕获范围。R1 发布卡的 TP32 超过上限，采用原 modelzoo 的 TP4×PP4 参考布局。Step/M3 模型卡的临时运行时补丁不执行，CI 检验本次构建的产品 wheel。发布卡中的历史 worker 退出不作豁免。

V4.1-1017 含 Engram CPU offload；DP1不创建Engram DP共享组，因此设置dp_shared_memory=false。使用`safetensors-load-strategy=prefetch`、每rank一个预读线程，把本地权重分片预读到系统页缓存；不改变权重格式、数值或版本。20上503 GiB主存、64 GiB shm、无限memlock已完成下述验证；运行前仍须核对可用主存。固定配置失败按失败保留，不自动切调度模式或减少卡数。

2026-11-11在20对V4.1做单次A/B：两组均TP4×PP4、DP1、无EP、V2异步调度/DSpark5，固定CI184 wheels及runtime digest `43e5ba10f1c57790bc5f4046260aef7aa3f48cd14cdedbc1c1058630de0d15fd`；启动前所选权重文件驻留页为0，两组编译缓存均新建。仅B增加上述预读参数，计时插桩相同。原lazy mmap的最慢rank模型加载343.134秒、完整启动678.340秒；预读分别283.391秒、602.237秒，缩短17.4%和11.2%。完整启动计时包含进程初始化、加载和预热，不含排队、安装wheel和实验专用页缓存清退；模型加载是16个rank的最大单rank耗时，不是纯磁盘读取时间。两组均20/20正常请求及正确答案，draft/accepted token增量分别1890/973、2085/1047；请求31.554秒、33.674秒。实验原始记录在20 `/home/zejun.cai/ci-v41-tppp-async-load-20261111`。这是V4.1单次对照结果，不代表其他模型或完整CI均已通过；预读参数仅用于该模型。

2026-11-12在20重启后复测：Step以发布配置0.6/0.95取样通过20/20。R1仅改取样时19/20，失败题反复输出正确FINAL直至4096 tokens截断；加单次FINAL提示后单模型20/20，但完整CI #192仍因一次只含reasoning、content为空的答复得19/20。R1改回确定性temperature=0、top_p=1，并明确要求先结束思考区、再在最终回答中只输出一次FINAL；使用CI #192固定wheels单模型复测通过20/20。题目和标准答案、输出上限、判分及无重试规则不变；完整CI仍以新提交归档为准。

Hy4在20重启后复测时，FULL_DECODE_ONLY仍在图捕获的分布式通信处报`Resource temporarily unavailable`，0/20请求；eager模式启动并完成20/20请求，MTP draft/accepted计数分别增长5688/3791。原解析器把并发题的`</think:opensource>FINAL:`识别为非JSON而得19/20，实际最终JSON为正确值。已增加该结束标签的边界解析和错误答案负例；最终CI结果仍以修复后PR构建归档为准。

## 权重与结果

权重唯一来源为ModelHub owner iluvatar-aiac-llmfp-inference的固定平台版本。源NFS仅用于复制，实际运行从20本地 `/opt/ci-models/<slug>/<平台版本>-<revision前12位>/` 只读加载。weights/ 下的独立清单包含全部发布文件的大小和SHA256，以及READY清单指纹；全部校验通过才发布本地目录。未就绪、版本不符、缺文件、软链接或NFS路径都拒绝运行。

11款权重已完成同步和SHA256校验；本次清单拆分保持展开内容、缓存路径及READY指纹不变，不要求重新复制。后续通过独立CI仓库的 `ci/sync_model_weights.py --suite <产品库>/tests/model_smoke/suite.json --job-dir <任务目录>` 准备缓存。

## 服务、请求和通过标准

runner 从 展开后的模型配置 取每款参数，执行 `vllm serve <本地路径> --served-model-name <本次唯一标识> --host 127.0.0.1 --port <空闲端口> ...`。同一进程组单独启动，HTTP端口按本次运行分配，配置中的master_port占位符也按本次替换；每款使用独立编译缓存。它清理继承的模型环境变量，锁定本地离线权重，最多可见16个逻辑 GPU。

1. 最多1800秒启动；`/v1/models` 必须包含本次唯一模型名，`/health` 正常。
2. 10题串行一遍，再以并发4执行同一组10题，共20次，无自动重试。默认temperature=0、top_p=1；Step按固定ModelHub generation_config使用temperature=0.6、top_p=0.95。seed固定、每请求最多4096 tokens/300秒。
3. **请求成功20/20且正确20/20**。必须返回本次模型名、一个非空最终答案、`finish_reason=stop`；超时、报错、截断、只有思考文本均失败。取最后一行末尾显式 `FINAL:` 标记后的完整答案（允许标记前有解释文字），数值按完整数字比较，文本按规范化后完整值比较，JSON按结构和值比较，不用“包含关键词”判正确。
4. 启用投机的3款还要求 `num_drafts`、`num_draft_tokens`、`num_accepted_tokens` 的运行增量均大于0，accepted不得超过draft；只有命令行开关不算生效。最多额外等20秒指标刷新，不重跑请求。
5. 结束时服务存活、健康检查通过，日志无 EngineDead/OOM/worker异常退出；随后清理本款进程组。清理失败则停止整组，不再启动下一款。

每款落盘 `launch.json`、`server.log`、`answers.jsonl`、`result.json`，投机款额外保存前后 metrics；整体保存 展开后的模型配置、summary、JUnit。没跑完的模型也在 JUnit 中记失败，不能跳过后显示全绿。

## PR触发

产品分支 `ci/model-smoke-*` 向main提PR时，CI读取本分支的suite.json、models JSON及其引用的weights清单，选择配置中的模型并给予720分钟总预算。显式Jenkins参数RUN_MODEL_SMOKE/MODEL_SMOKE_MODELS仍可用于开启或选择子集；普通分支默认不开启真实模型服务。

CI不再保存模型参数副本。选模结果包含所有模型配置的展开指纹；实际运行归档展开后的manifest.json、cases.json、每模型launch/answers/result/server日志及整体summary/JUnit，构建来源记录产品源码SHA。任何模型或pytest失败都使构建失败；模型清理失败后不继续启动下一款。整个运行受同一整机GPU锁保护。

## 同一CI内的模型并行

默认最多并行三款模型，按每款logical_gpus分配互不重叠的CUDA_VISIBLE_DEVICES，整组最多16个逻辑GPU。suite优先排列Qwen3.6-35B-A3B和Qwen3.8-27B，两款各TP4共占8卡，可与后续一款8卡模型同时运行；Qwen3.8-Flash-Next仍为TP8。8卡模型可两两并行，16卡模型独占整机；调度可先运行后面能够放入空闲卡的模型，结果仍按suite中的模型身份归档。使用`--max-parallel-models 1`可做串行对照。不同CI之间仍共用原整机锁，pytest与模型阶段仍串行。

每款独立服务名、HTTP/master端口、进程组和缓存目录；只有服务清理结束才归还GPU配额。清理失败或套件收到终止信号时停止派发新模型，并终止本runner已创建的其他模型进程组；未完成项保持失败。普通模型判分失败仍保留结果，继续验证其余模型，不重试请求。

每模型result.json增加allocated_devices、started_at/finished_at及timings，区分preflight、startup、requests、validation和cleanup耗时。11款权重版本、20/20答案判定与投机指标门槛保持不变；上述两款Qwen从TP8改为TP4。TP4兼容性和显存峰值须通过真卡验证。并行调度会竞争本机磁盘与主存，净加速收益以真卡运行结果为准，不把CPU并发测试当作性能结论。

## 冒烟问题与容量修正

数值/文本题仅要求对应类型的最终值，不再附加JSON指令；JSON提取题分别说明输入记录和输出schema，明确只返回name/age、不返回city。问题内容和标准答案未变。最终答案可以在解释文字后的`FINAL:`标记之后；`</think>`及`</think:opensource>`后的最终答案也可识别，但仍完整比较数值、文本或JSON；额外字段、错误值、只有reasoning、截断与空正文仍失败。Hy4将max_model_len从17408调整为8192，适配20上实测约1.17 GiB可用KV缓存；仍保留每请求4096输出tokens、并发4、MTP3和20/20要求。

组织更名仅改变owner。已有本地缓存保留原始READY回执：runner只接受与当前固定版本、完整文件SHA256清单及原owner重新计算的指纹完全相符的旧回执；不修改权重或历史校验记录。新同步使用新owner和新指纹。

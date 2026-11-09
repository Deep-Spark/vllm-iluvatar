# FlyDSL 跨 provider Benchmark 使用说明

在同一套 workload 上对比 **FlyDSL（ixaite）** 与 **baseline（ixformer / IXF）**。
生产路由开关（`VLLM_ILU_KERNEL_PROVIDER`）仅作文末对照。

入口：在 **vllm-iluvatar 仓库根目录**执行 `python -m tests.ixaite.benchmarks`。

---

## 1. Benchmark 做什么

统一 harness 对比两端实现：

| 端 | 来源 | 例 |
|---|---|---|
| **baseline** | ixformer / 社区 FLA / 插件 Triton（对应生产路径） | `IXF.linear`、FLA `rmsnorm_fn` |
| **ixaite** | ixaite 公共 API（FlyDSL） | `ixaite.linear`、`ixaite.chunk_gated_delta_rule` |

每次运行固定顺序：

1. **正确性**（eager）：两端各跑一遍，按 Suite 容差 `assert_close`。
2. **计时**（可选）：GPU event 采 median latency；默认同时报 **eager** 与 **CUDA Graph replay**。
3. **报告**：写入 `tests/ixaite/benchmarks/reports/` 的 JSON + CSV（可用 `--no-write` 跳过）。
4. **门禁**（`--gate`）：失败则进程退出码 `1`。

Bench 直接取得两端 callable，不依赖服务的后端选择。报告用于评估正确性和性能；
服务在启动时固定一个后端，forward 中不根据计时结果重新选择。

### 1.1 Suite 与 Case

- **Suite** = 一个 provider ABI（例如 `linear`、`silu_and_mul`）。
- **Case** = call-site + 模型 shape（例如 `attn_qkv_proj@qwen35_9b;T=128`）。
  `attn_qkv_proj` 是 `linear` Suite 下的 Case，不是独立框架边界。

当前已注册 Suite（`python -m tests.ixaite.benchmarks --list`）：

| Suite | baseline | ixaite | 首版 Case / 模型 |
|---|---|---|---|
| `silu_and_mul` | IXF silu | `ixaite.silu_and_mul` | `qwen35_9b`，`D=12288`；`qwen36_27b` L2 TP2 `D=8704`（e2e serve TP=4）；`qwen36_27b_w8a8_tp2` / `tp4`，`D=8704` / `4352` |
| `linear` | `IXF.linear` | `ixaite.linear` | `qwen35_9b`（TP=1）及 `qwen35_9b_tp2` / `qwen35_9b_tp4`（shard-local N/K）；`qwen36_27b` BF16 L2 TP2（GDN + MLP + full-attn qkv/o + lm_head，e2e serve TP=4）；`qwen36_27b_w8a8_tp2` / `qwen36_27b_w8a8_tp4`（GDN + lm_head，GDN-out K=6144/TP） |
| `embedding_gather` | `pytorch.F.embedding` | `ixaite.embedding_gather` | `qwen35_9b`（`V=248320,H=4096`）及 `qwen36_27b_w8a8_tp2` / `qwen36_27b_w8a8_tp4`（`V=248320,H=5120`），T=1/8/32/128/256/512/1024/2048 |
| `w8a8` | `IXF.w8a8` | `ixaite.w8a8` | `qwen35_9b_w8a8`（W8A8 量化 checkpoint），`attn_qkv_proj`（`K=4096,N=10240`）；`qwen36_27b_w8a8_tp2` / `tp4` 覆盖 full-attn qkv/o 与 MLP gate_up/down（qkv `N=7168/3584`） |
| `scaled_int8_quant` | `IXF.dynamic_scaled_int8_quant` | `ixaite.scaled_int8_quant` | `qwen35_9b_w8a8`，激活 `in_features` `K=4096/12288`；`qwen36_27b_w8a8_tp2` / `tp4` 为 `5120/3072/8704` 与 `5120/1536/4352` |
| `scaled_int8_quant_asym` | `torch.dynamic_asym_int8_quant` | `ixaite.scaled_int8_quant[asym]` | 同上；基线用 torch eager，ixformer 无非对称 INT8 量化 |
| `activation_dynamic_scaled_int8` | `IXF.activation_dynamic_scaled_int8` | `ixaite.ops.activation_dynamic_scaled_int8` | `qwen3_30b_a3b_w4a8` 路由 expert swiglu，`H=768`，`output_format` 0/1 |
| `moe_w4a8_group_gemm` | `IXF.moe_w4a8_group_gemm` | `ixaite.ops.moe_w4a8_group_gemm` | `qwen3_30b_a3b_w4a8` prefill `w13` `N=1536,K=2048` 与 `down` `N=2048,K=768` |
| `moe_w4a8_group_gemv` | `IXF.moe_w4a8_group_gemm`（同一 TN 数值） | `ixaite.ops.moe_w4a8_group_gemv` | 同上 checkpoint 的 decode `w13` / `down`；ixaite GEMV 对齐 group_gemm，不对齐 IXF `moe_w4a8_group_gemv` |
| `moe_compute_token_index` | `IXF.moe_compute_token_index[_ep|_for_decode]` | `ixaite.ops.moe_compute_token_index` | `qwen3_30b_a3b_w4a8`：128 experts / topk 8，plain+EP+decode；`deepseek_v4_w4a8` 覆盖 TP=1 非 EP 的 plain/decode |
| `moe_expand_input_dynamic_scaled_int8` | `IXF.moe_expand_input_dynamic_scaled_int8` | `ixaite.ops.moe_expand_input_dynamic_scaled_int8` | `qwen3_30b_a3b_w4a8` hidden 2048 / topk 8 |
| `moe_fused_topk_bias` | `IXF.moe_fused_topk_bias` | `ixaite.ops.moe_fused_topk_bias` | `qwen3_30b_a3b_w4a8` E=128 / topk 8 softmax |
| `moe_output_reduce_sum` | `IXF.moe_output_reduce_sum` | `ixaite.ops.moe_output_reduce_sum` | `qwen3_30b_a3b_w4a8` hidden 2048 / topk 8 |
| `rms_norm` | `IXF.rms_norm` | `ixaite.rms_norm` | `qwen3_8b` 无 residual 的 input / final norm，`H=4096` |
| `rms_norm_qk` | `IXF.rms_norm_qk` | `ixaite.rms_norm_qk` | `qwen3_8b` 融合 attention q/k norm，`D=128`，Hq=32 / Hkv=8 |
| `rms_norm_gated` | 社区 FLA Triton `rmsnorm_fn` | `ixaite.rms_norm_gated` | `qwen35_9b` GDN `[T,32,128]`；`qwen36_27b_w8a8_tp2` / `tp4` 是 `[T,24/12,128]`（Dv，不是 hidden `5120`） |
| `fused_add_rms_norm` | `IXF.residual_rms_norm` | `ixaite.fused_add_rms_norm` | `qwen35_9b`，`H=4096` residual 路径；`qwen36_27b` / W8A8 TP2/TP4 `H=5120` |
| `gemma_rms_norm` | `IXF.gemma_rms_norm` | `ixaite.gemma_rms_norm` | `qwen35_9b` GemmaRMSNorm 2D 路径（含 final norm 无 residual），`H=4096`；`qwen36_27b` / W8A8 TP2/TP4 `H=5120` |
| `layer_norm` | `IXF.layer_norm` | `ixaite.layer_norm` | `qwen35_9b` ViT block norm1/norm2 与 merger 入口，`H=1152`，T=784/3136/12544（448²/896² 每图行数，以及 4 图并发）|
| `video_evs` | 社区 PyTorch EVS | `ixaite.video_evs` | `qwen35_9b_video` merger 后的 stable retention mask，主输入为 `[392,3584]` |
| `vit_qkv_layout` | eager permute split | `ixaite.vit_qkv_layout` | `qwen35_9b` ViT packed QKV，`D=72`，H=16/8/4，S=784/3136 |
| `vit_qkv_rope` | eager split + NeoX RoPE | `ixaite.vit_qkv_rope` | `qwen35_9b` / `qwen36_27b_vision` ViT packed QKV + partial-0.5 NeoX，`D=72`，H=16/8/4，S=784/3136 |
| `causal_conv1d_fn` | 插件 Triton prefill | `ixaite.causal_conv1d_fn` | `qwen35_9b` GDN，`dim=8192,width=4`；`qwen36_27b_w8a8_tp2` / `qwen36_27b_w8a8_tp4` 的 GDN 列切分片 `dim=5120` / `2560`，`width=4` |
| `causal_conv1d_update` | 插件 Triton decode | `ixaite.causal_conv1d_update` | 同上（`8192` / `5120` / `2560`，`width=4`） |
| `fused_post_conv_prep` | 插件 rearrange + Triton gating + l2norm | `ixaite.fused_post_conv_prep` | `qwen35_9b` TP=1，`Hk=16,Hv=32,D=128`；`qwen36_27b_w8a8_tp2` / `tp4` 为 `Hk=8/4,Hv=24/12` |
| `chunk_gated_delta_rule` | ixformer infer chunk | `ixaite.chunk_gated_delta_rule` | `qwen35_9b` TP=1 dense 512 / packed 512+1536；27B 同 lengths，头数随 TP 为 `8/24`、`4/12` |
| `gdn_packed_decode` | 社区 FLA Triton packed decode | `ixaite.gdn_packed_decode` | `qwen35_9b` TP=1，`Hk=16,Hv=32,D=128`；27B TP2/TP4 为 `8/24`、`4/12` |
| `gdn_spec_delta_update` | ixformer infer recurrent | `ixaite.gdn_spec_delta_update` | `qwen35_9b` TP=1，batch 1/8、每序列 2/4 token；27B W8A8 L2 在 `qwen36_27b_w8a8_tp2` / `tp4`，e2e serve 走 `*_ngram` |
| `attention` | `ixformer.contrib.vllm_flash_attn` | `ixaite.flash_attn` | `qwen35_9b` GQA4 `16/4`；27B TP2/TP4 GQA6 `12/2`、`6/1`；35B TP2/TP4 `8/1`（GQA8）、`4/1`（GQA4，两个 KV head 被复制）；`D=256,page=16` |
| `reshape_and_cache_flash` | ixformer HND cache writer | `ixaite.reshape_and_cache_flash` | `qwen35_9b` TP=1，BF16 `Hkv=4/2/1,D=256,page=16`；27B TP2/TP4 `Hkv=2/1`；35B TP2/TP4 `Hkv=1/1` |
| `ds4_fused_qnorm_rope_kv_insert` | `IXF.ds4_fused_qnorm_rope_kv_insert` | `ixaite.ds4_fused_qnorm_rope_kv_insert` | `deepseek_v4_w4a8` BF16 SWA，`H=8,D=512,rotary=64,page=64`（0731 TP8 本地头数） |
| `ds4_inv_rope` | `IXF.ds4_inv_rope` | `ixaite.ds4_inv_rope` | `deepseek_v4_w4a8` GPT-J 逆 RoPE，`H=8,D=512,rotary=64` |
| `ds4_mhc_pre` | `IXF.ds4_mhc_pre_v2` | `ixaite.ds4_mhc_pre_v2` | `deepseek_v4_w4a8` MHC v2，`n_hc=4,hidden=4096` |
| `ds4_mhc_post` | `IXF.ds4_mhc_post` | `ixaite.ds4_mhc_post` | `deepseek_v4_w4a8` MHC post mix，`n_hc=4,hidden=4096` |
| `ds4_hc_head` | `IXF.ds4_hc_head` | `ixaite.ds4_hc_head` | `deepseek_v4_w4a8` hc_head collapse，`n_hc=4,hidden=4096` |
| `ds4_compressor_save_partial_states` | `IXF.ds4_compressor_save_partial_states` | `ixaite.ds4_compressor_save_partial_states` | `deepseek_v4_w4a8` C4A fp32 state write，`D=512,cr=4` |
| `ds4_compressor_pool_norm_rope_insert` | `IXF.ds4_compressor_pool_norm_rope_insert` | `ixaite.ds4_compressor_pool_norm_rope_insert` | `deepseek_v4_w4a8` C4A pool+RMS+GPT-J，`D=512,cr=4` |
| `gemm_bf16_bf16_fp32` | `IXF.gemm_bf16_bf16_fp32` | `ixaite.gemm_bf16_bf16_fp32` | `deepseek_v4_w4a8` C4A compressor GEMM，`N=2048,K=4096` TN bf16->fp32 |
| `marlin_w8a16` | `IXF.marlin_w8a16` | `ixaite.marlin_w8a16` | `deepseek_v4_w4a8` wo_a `N=1024,K=4096` B=1/8；W_UK `k16n16_grouped_n` `N=512,K=448` B=8/64 |
| `reshape_and_cache_with_scale` | 插件 CUDA `_C.reshape_and_cache_with_scale` | `ixaite.reshape_and_cache_with_scale` | `qwen35_9b_kvint8`（`KVINT8`），`Hkv=4,D=256,page=16` |
| `reshape_and_cache_mix` | 插件 CUDA `_C.reshape_and_cache_mix` | `ixaite.reshape_and_cache_mix` | `qwen35_9b_kint8`（`KINT8`），`Hkv=4,D=256,page=16` |
| `fused_qk_rmsnorm_rope_gate` | 插件 CUDA `_C.fused_qk_rmsnorm_rope_gate` | `ixaite.fused_qk_rmsnorm_rope_gate` | `qwen35_9b` TP1 `16/4`；27B TP2/TP4 `12/2`、`6/1`；35B TP2/TP4 `8/1`、`4/1`；均为 BF16 `D=256,rotary=64` |
| `fused_qk_rmsnorm_gate` | 插件 CUDA `_C.fused_qk_rmsnorm_gate` | `ixaite.fused_qk_rmsnorm_gate` | `qwen35_9b` TP=1，BF16 `Hq=16,Hkv=4,D=256` |
| `attn_output_sigmoid_gate` | Torch `attn_output * sigmoid(gate)` | `ixaite.attn_output_sigmoid_gate` | `qwen35_9b` TP=1，BF16 `D=4096`（`16*256`）；27B TP2/TP4 `D=3072/1536`；35B TP2/TP4 `D=2048/1024`（均为 `Hq*256`，不是 hidden） |
| `distributed` | `torch.distributed.all_reduce`（IXCCL） | `ixaite.distributed.all_reduce` | `qwen36_27b` / `qwen36_27b_w8a8_tp2` / `tp4` residual `[T,5120]`；单进程 L2 是 world-size 1；TP 路径由这三条 e2e 覆盖 |

`mlp_down_proj` 的 shape 在 `tests/ixaite/benchmarks/models/qwen35_9b.yaml` 的
`l2.linear.mlp_down_proj` 中可做微测。服务使用整体后端选择；
ixaite 目前拒绝尚未验证的固定形状，不再将其转给 IXF。

`linear` 的默认 T 集合见 `tests/ixaite/benchmarks/core/profiles.py`：完整 CUDA Graph
capture 表（1/2/4 + 8..248 步长 8 + 256，共 35 点）加 prefill 锚点
（512/1024/2048）。非 2 的幂的 capture 尺寸落运行时 safe-default pad 路径，
必须全部实测，不能靠 2 的幂插值。

其余适用动态 token/batch 的 Suite 使用精简锚点
`1/8/128/256/512/2048`；decode-only conv update 到 `256`，prefill-only conv
从 `256` 开始。Chunk 使用生产 dense/packed case，不展开通用 T 笛卡尔积。
`causal_conv1d_fn` 是 prefill-only 路径，Suite 只测 eager；显式请求
`--timing cuda-graph` 会报错，默认 `both` 会跳过 graph。
两个 int8 cache-writer Suite 的首版性能口径也是 eager，运行时传
`--timing eager`；`--gate` 只检查正确性和稳定性，不设置最低 speedup。

GDN conv 的 `dim` 是列切分片 `(2*Hk*Dk + Hv*Dv) / TP`，不是 hidden/TP。
Qwen3.6-27B 全局 `Hk=16, Hv=48, Dk=Dv=128`，即 `10240`，所以 TP2 是
`5120`、TP4 是 **`2560`**；写成 9B TP4 的 `2048` 是错的。同一套切片下
prep / chunk / packed_decode 的头数是 TP2 `8/24`、TP4 `4/12`，不是 9B
的 `16/32` 或其 TP4 `4/8`。`rms_norm_gated` 是 `[T, Hv, 128]`。

`attn_output_sigmoid_gate` 乘在 `o_proj` 之前的 `attn_output` 上，last
dim 是 TP 本地 `Hq * head_dim`，不是 hidden。9B 的 `4096` 是 `16*256`，
刚好等于 hidden；27B hidden 是 `5120`，serve 宽度是 TP2 `12*256=3072`、
TP4 `6*256=1536`。

会原地改 recurrent 状态的 decode Case（`causal_conv1d_update` 的 `x` /
`conv_state`，`gdn_packed_decode` / `gdn_spec_delta_update` 的 SSM
`initial_state`）在连续 replay 里会把每一步的输出喂给下一步。BF16 误差
叠上去之后，post-timing 的 `check` 就会对不上。Case 用
`prepare_cuda_graph_replay` 在计时后复位输入，最后一次 replay 才是一个
定义明确的单步，可以与 baseline 对齐。

### 1.2 计时模式

| `--timing` | 报告里 `mode` 列 | 计什么 |
|---|---|---|
| `both`（默认） | `bench` + `bench_cudagraph` | eager 启动 + `CUDAGraph.replay()` |
| `eager` | `bench` | eager kernel launch |
| `cuda-graph` | `bench_cudagraph` | warmup+capture 之后的 replay（capture / JIT 不计时） |

正确性始终走 eager。`--check-only`：只做正确性，跳过全部计时（忽略 `--timing`）。

Speedup 定义：

```text
speedup = median_ms(baseline) / median_ms(ixaite)
# >1 => ixaite 更快；0.1 => ixaite 约慢 10x
```

---

## 2. 环境准备

需已安装并可 import：

- `torch`（Iluvatar / CoreX）
- `ixformer`
- `ixaite`（editable 或 wheel 均可）
- FlyDSL（Iluvatar 后端；`FLYDSL_COMPILE_BACKEND=iluvatar`）

```bash
cd /path/to/vllm-iluvatar

# 若未做 editable install，把本仓加入 PYTHONPATH 即可
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export FLYDSL_DUMP_IR=0   # bench 时关掉 IR dump

# 需要 Iluvatar GPU；ixaite.initialize() 需匹配当前 arch（如 ivcore11）
python -m tests.ixaite.benchmarks --list
```

---

## 3. 使用方法与命令

```bash
python -m tests.ixaite.benchmarks [选项...]
python -m tests.ixaite.benchmarks --help    # 完整参数、默认值与示例
```

### 3.1 发现

```bash
python -m tests.ixaite.benchmarks --list
```

### 3.2 正确性（开发 / smoke）

```bash
# silu：便宜正确性 + 门禁（只验 correct）
python -m tests.ixaite.benchmarks --suite silu_and_mul --model qwen35_9b --check-only --gate

# linear：ixaite 路径 call-site 正确性（qkv / o / gate_up / lm_head）
python -m tests.ixaite.benchmarks --suite linear --case attn_qkv_proj --model qwen35_9b --check-only
python -m tests.ixaite.benchmarks --suite linear --case attn_o_proj --model qwen35_9b --check-only
python -m tests.ixaite.benchmarks --suite linear --case mlp_gate_up_proj --model qwen35_9b --check-only
python -m tests.ixaite.benchmarks --suite linear --case lm_head --model qwen35_9b --check-only

# 正确性 + gate（失败 exit 1）
python -m tests.ixaite.benchmarks --suite linear --case attn_qkv_proj --model qwen35_9b \
  --check-only --gate
python -m tests.ixaite.benchmarks --suite linear --case attn_o_proj --model qwen35_9b \
  --check-only --gate
python -m tests.ixaite.benchmarks --suite linear --case mlp_gate_up_proj --model qwen35_9b \
  --check-only --gate
python -m tests.ixaite.benchmarks --suite linear --case lm_head --model qwen35_9b \
  --check-only --gate
```

### 3.3 完整计时 + PR 门禁

```bash
# 默认：正确性 → eager + CUDA Graph 计时 → 写报告 → gate
python -m tests.ixaite.benchmarks --suite silu_and_mul --model qwen35_9b --gate

python -m tests.ixaite.benchmarks --suite linear --model qwen35_9b --gate

# 只跑一个 call-site（ixaite 路径）
python -m tests.ixaite.benchmarks --suite linear --case attn_qkv_proj --model qwen35_9b --gate
python -m tests.ixaite.benchmarks --suite linear --case attn_o_proj --model qwen35_9b --gate
python -m tests.ixaite.benchmarks --suite linear --case mlp_gate_up_proj --model qwen35_9b --gate
python -m tests.ixaite.benchmarks --suite linear --case lm_head --model qwen35_9b --gate
```

### 3.4 只跑一种计时

```bash
python -m tests.ixaite.benchmarks --suite silu_and_mul --model qwen35_9b --timing eager --gate
python -m tests.ixaite.benchmarks --suite silu_and_mul --model qwen35_9b --timing cuda-graph --gate
```

### 3.5 稳定性与采样

```bash
# 默认：warmup=20, repeats=100；相对稳定性 max 在 median 的 3% 内
python -m tests.ixaite.benchmarks --suite silu_and_mul --model qwen35_9b --gate

# 放宽相对稳定性到 5%
python -m tests.ixaite.benchmarks --suite silu_and_mul --model qwen35_9b --stability-pct 5 --gate

# 额外允许绝对抖动（微秒）；与 pct 同时设时取更大 slack
python -m tests.ixaite.benchmarks --suite silu_and_mul --model qwen35_9b --stability-us 5 --gate

# 自定义采样
python -m tests.ixaite.benchmarks --suite linear --model qwen35_9b --warmup 10 --repeats 50 --gate
```

### 3.6 Case / dtype / 报告

```bash
# 按 case_id 子串过滤（如 rows=8、attn_qkv_proj、T=128）
python -m tests.ixaite.benchmarks --suite silu_and_mul --case rows=8 --no-write
python -m tests.ixaite.benchmarks --suite linear --case T=128 --check-only

# dtype（默认 bfloat16）
python -m tests.ixaite.benchmarks --suite linear --model qwen35_9b --dtype float16 --check-only

# linear：扩到完整 CUDA Graph T 表
python -m tests.ixaite.benchmarks --suite linear --model qwen35_9b --full-profile --check-only

# 指定报告目录 / 只打印摘要不落盘
python -m tests.ixaite.benchmarks --suite linear --model qwen35_9b --out tests/ixaite/benchmarks/reports --gate
python -m tests.ixaite.benchmarks --suite linear --model qwen35_9b --gate --no-write
```

报告默认路径：`tests/ixaite/benchmarks/reports/`（gitignored），文件名形如：

```text
{suite}_{model}_{UTC戳}_{eager_cudagraph|check}.json
{suite}_{model}_{UTC戳}_{eager_cudagraph|check}.csv
```

---

## 4. Gate 规则（首版）

启用 `--gate` 时：

| 模式 | 要求 |
|---|---|
| `--check-only --gate` | 每个 Case **correct** |
| 完整 `--gate` | correct + **stable** + `speedup >= suite.gate_min_speedup()`（通常 `1.0`） |

稳定性：

- 默认：`max(samples) <= median * (1 + stability_pct/100)`，`--stability-pct` 默认 `3`。
- 可选 `--stability-us N`：另允许 `max <= median + N/1000` ms；与 pct 同时设时取更大 slack。

测量结果应附上命令、设备、报告路径和未通过的项目。算子 gate 通过与模型
E2E 通过分别记录，不能将部分形状通过当作整个模型已支持。

---

## 5. 与服务后端的关系

`VLLM_ILU_KERNEL_PROVIDER` 只接受 `baseline|ixaite`，服务默认 `baseline`。
原目录保留 baseline；ixaite 实现和注册位于
`vllm_iluvatar/backends/ixaite`。算子接口、层、量化对象、Worker 和通信对象
在初始化时一起选定，加载与执行期间不跨后端回退。

```bash
export VLLM_ILU_KERNEL_PROVIDER=ixaite
```

后端来源断言只在测试启动入口中启用，profiler 用于单独的功能验收。
正常性能测量不带断言或 profiler。模型支持情况需要按具体配置进行
E2E 验证。

---

## 6. 新增 Suite

接入插件的 ixaite op **必须**同时加 harness。`tests/ixaite/benchmarks/test_plugin_op_benchmark_suites.py`
会扫描导出算子表、初始化绑定和相关生命周期 loader；
缺 mapping 或未注册 Suite 时检查失败。模型依赖检查与实际支持的算子分别记录。

1. 新增 `suites/<op>.py`，实现 `resolve_providers` / `iter_cases`，并 `register(...)`。
2. 在 `suites/__init__.py` import 以注册。
3. 把 plugin op 名写进 `tests/ixaite/benchmarks/core/plugin_op_suites.py` 的 `PLUGIN_OP_SUITES`。
4. 在 `tests/ixaite/benchmarks/models/<model>.yaml` 的 `l2.<suite>` 中登记模型 shape；
   Suite 从统一模型表读取，不在 `suites/*.py` 内硬编码模型名。
5. 把 plugin op 名写入至少一个 `tests/ixaite/benchmarks/models/<model>.yaml` 的
   `e2e.ops`；该模型的 serve 路径必须真正打到这个 op。现有 e2e 模型打不到
   时，新增或扩展 profile，不要只挂 L2。
6. 先 `--check-only`，再完整 `--gate`；报告放进 PR。
   `tests/ixaite/benchmarks/test_plugin_op_benchmark_suites.py` 会同时检查 L2 suite 与 e2e.ops。

---

## 7. 整模 e2e：FlyDSL vs baseline

e2e 是独立的重型入口，不会被 `python -m tests.ixaite.benchmarks` 或 `--gate` 隐式触发：

```bash
python -m tests.ixaite.benchmarks.e2e --help
python -m tests.ixaite.benchmarks.e2e --list-models
python -m tests.ixaite.benchmarks.e2e \
  --model qwen35_9b \
  --model-path /path/to/Qwen3.5-9B \
  --out tests/ixaite/benchmarks/e2e-runs/qwen35-run
```

默认一条命令先后启动两次独立 `vllm serve`，对同一组 workload 测量：

1. `baseline`：非 FlyDSL 生产实现（ixformer / 插件 Triton / 插件 CUDA）。
2. `ixaite`：已接入算子使用 ixaite/FlyDSL；缺实现时启动失败。

用户不需要设置 `VLLM_ILU_KERNEL_PROVIDER`。e2e runner 分别把
`baseline` 和 `ixaite` 注入对应的 serve 子进程。插件在进程启动时选定实现，
所以两个 provider 必须分别启动服务。调试单侧时可加
`--provider baseline` 或 `--provider ixaite`。这是两个合法值；省略参数时对比两端，
无需设置 `both`。服务自身的默认值仍是 `baseline`。

每个 provider、每个 workload 都通过 `vllm bench serve --num-warmups`
发送不计入结果的预热请求。权重路径、端口和设备不写入模型 profile：

```bash
# --device 会作为子进程 CUDA_VISIBLE_DEVICES；不传则继承当前环境。
python -m tests.ixaite.benchmarks.e2e \
  --model qwen35_9b \
  --model-path /path/to/Qwen3.5-9B \
  --device 0 \
  --port 8000
```

`--dry-run` 只校验 profile 并打印完整 serve/bench 命令，不启动服务、不写文件。

可选的 `e2e.correctness` 探针在吞吐测量之后跑，不计入 latency。两种 kind：

- `video_greedy_exact`：两个 provider 各发一次同一条 greedy 请求，输出必须逐字节相同。挂在有视觉塔的 profile 上（目前 `qwen35_9b_video`）。
- `tp_graph_vs_eager`：每个 provider 和自己比。同一条 greedy 请求在 graph 服务上重复 `repeats` 次必须相同，再和追加 `--enforce-eager` 的参照服务比。挂在会走 tensor-parallel `all_reduce` 的 profile 上（目前 `qwen35_9b_tp2`）；tp=1 时集合通信是空操作，挂上去没有意义。

### 7.1 输出

默认目录为 `tests/ixaite/benchmarks/e2e-runs/<UTC时间>-<model>/`：

```text
manifest.json
compare.json
metrics.csv
summary.txt
phases/
  baseline/
    server.log
    <workload>.json
    <workload>.log
  ixaite/
    server.log
    <workload>.json
    <workload>.log
```

`compare.json` 使用 schema version 2，移除了旧的 `routes` 字段；
`metrics.csv` 保持原指标列。两者均为机器可读数据，不生成 HTML。
吞吐 speedup 为 `ixaite / baseline`，延迟 speedup 为
`baseline / ixaite`，因此两者都以 `> 1` 表示 ixaite 更好。

runner 校验请求完成数、失败数和输出 token 总数，并保留服务日志、原始结果和
启动环境。`e2e.ops` 登记预期覆盖范围；日志不再用来证明算子执行来源。
严格功能测试通过测试启动入口安装 ixformer 调用断言，再按需采集 profiler
证据；这两项检查不进入正常性能测量。runner 保留用户的日志级别及遗留开关，
冲突由服务的统一模式校验报告。

### 7.2 新增模型

在 `tests/ixaite/benchmarks/models/` 增加一个与 slug 同名的 YAML profile，即会自动出现在
L2 和 e2e 的 `--list-models` 中。profile 同时包含：

- `l2`：每个 Suite 的模型 shape。只需覆盖打算对该模型运行的 Suite；
  门禁要求每个已注册 Suite 至少出现在一个模型的 `l2` 里。
- `e2e`：可移植的 serve 参数、`ops`（plugin op 名，serve 路径会打到的集合）、
  环境开关、warmup 和 workload。新接入的 plugin op 必须出现在至少一个模型的
  `e2e.ops` 里。默认 BF16 serve 打不到的 op 应放到独立变体 profile，不要挂在
  `qwen35_9b.yaml` 上：量化 GEMM 用 `qwen35_9b_w8a8.yaml`，int8 KV writer 用
  `qwen35_9b_kint8.yaml` / `qwen35_9b_kvint8.yaml`。Qwen3.6-27B W8A8 默认
  `qwen36_27b_w8a8_tp2` / `tp4` 对齐 ixformer（不开 ngram）；`gdn_spec_delta_update`
  走 `qwen36_27b_w8a8_tp2_ngram` / `qwen36_27b_w8a8_tp4_ngram`。
  `e2e.ops` 记录对应配置，默认 BF16 KV 不覆盖 int8 writer。`qwen35_9b_kvint8` 额外需要
  宿主环境里的 `VLLM_ILU_FLASH_ATTN_V_SCALE_PATH`（校准文件路径不写入
  profile）。视觉塔上的 op 同理走 `qwen35_9b_vision.yaml`：它开 `image:1`，
  workload 用 `dataset: random-mm` 加 `mm.bucket_config` 真发图，serve forward
  才会进 ViT。

不要在 profile 中加入权重绝对路径、`CUDA_VISIBLE_DEVICES`、本机 venv 或端口。
新增模型不需要修改 argparse 分支，也不需要在各个 Suite 重复添加模型名。

### 7.3 INT8 缓存接 baseline Attention 的诊断对照

`tests.ixaite.benchmarks.e2e.int8_cache_compare` 用同一个 BF16 MHA/GQA 模型，对比原生缓存写入与共享 ixaite 缓存写入。两边的 Q 量化、prefill/decode Attention 和其余模型算子均使用 baseline；`shared` 阶段显式选择测试用 `CacheReadWorker`。该入口仅用于诊断，不表示生产 ixaite 后端已支持原生 INT8 Attention。

共享路径先调用 `Int8KVCache.write`，随后 baseline Attention 只读取缓存；原生路径保留 decode 融合写入。测量包含独立 writer 的 kernel 启动开销，用于比较完整缓存接入方式的端到端表现。

在已初始化的测试环境中，从仓库根目录运行，并选择空闲 GPU 和新的输出目录：

```bash
export PYTHONPATH=/path/to/ixaite/src:/path/to/vllm-iluvatar:${PYTHONPATH:-}
python -m pytest tests/ixaite/attention/test_int8_cache_baseline_bridge.py -q

for phase in calibrate KINT8-baseline KINT8-shared KVINT8-baseline KVINT8-shared; do
  python -m tests.ixaite.benchmarks.e2e.int8_cache_compare \
    --model-path /path/to/model \
    --output tests/ixaite/benchmarks/e2e-runs/int8-cache-comparison \
    --device 2 --phase "$phase" || break
done
```

`calibrate` 使用 baseline 的静态 V 标定流程，验证输出文件的层数、形状及 scale 是否为有限正数；KVINT8 的两个阶段复用同一文件。该小样本标定仅供诊断，不代表量化质量验收。

对照时应从服务日志核对同一量化模式下两条路径的实际缓存容量。`--eager` 可关闭 CUDA Graph，`--tp` 设置张量并行度；其他参数见 `--help`。

每个阶段保存 `manifest.json`、服务端/客户端日志、原始 benchmark JSON 和 `probes.json`。探针检查输出长度及同一路径内重复请求的一致性；跨路径输出需要另行比较。吞吐测试和重复请求稳定性不能代替整模精度验收。

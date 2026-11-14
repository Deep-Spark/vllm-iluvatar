# DSV4.1 ixmm fused MoE

`VLLM_ILU_MOE_IXMM_FUSION=1` enables the fused target expert FFN on **D** for
both decode and eager replay. Default: `0`. Leave it off on P.
It is independent of `VLLM_ILU_DP_PREFILL_ELIGIBLE_QUORUM`.

## Build

Use the matching `tencent_dsv41_ixmm` ixformer branch. Its
`third_party/ixmm` gitlink pins the kernel source, including the compact replay
entry; do not replace it with the current ixmm master or an unrelated wheel.

In the ixformer checkout, initialize submodules and build using the existing
CoreX environment and the normal project installation procedure:

```bash
git submodule update --init --recursive
bash third_party/ixmm/scripts/install_ixmm.sh --arch ivcore11
bash scripts/build_ixformer.sh
bash scripts/install_ixformer.sh
```

ixformer automatically discovers the installed ixmm package, as for its existing
ixmm backend. On MR (`ivcore11`) it checks the fused MoE APIs and links
`libixmm.so`. No additional ixformer build switch is needed. Kernels are compiled
in ixmm, not duplicated into ixformer. An old/missing ixmm package disables the
bridge at build time; requesting fusion then reports an installation error.

## D configuration

The supported configurations are TP1×DP16/EP16 and TP2×DP8/EP16, BF16,
W4A8 NN V2 per-channel
weights, 384 routed experts, 24 local experts, hidden size 5120, intermediate
size 2304, SwiGLU clamp 10. Target layer 0–39 is converted once during loading;
draft weights remain unchanged. For a TN checkpoint, keep the existing
TN-to-NN runtime conversion enabled. The extra fused permutation happens after
that conversion. Disabling this switch requires restarting from the original
checkpoint; it is not a live toggle.

```bash
# D only. Enable fused FC1 + activation/quantization + FC2.
export VLLM_ILU_MOE_IXMM_FUSION=1

# Existing NVSM direct-store protocol prerequisites.
export IXFORMER_NVSM_W2_DIRECT_COMBINE=1
export IXFORMER_NVSM_SINGLE_ACQUIRE=1
export IXFORMER_NVSM_COMBINE_SINGLE_ACQUIRE=1
export IXFORMER_NVSM_COMBINE_TOKEN_STRIPES=4
export IXFORMER_NVSM_DISPATCH_VARIANT=batch_split
export IXFORMER_NVSM_DISPATCH_PUSH_PAYLOAD=1
export IXFORMER_MOE_NN_COMPACT_TASKS=1
```

Keep the existing **NVSM** backend selection and **BF16 combine**. The direct
chain supports target top-k 6 and actual registered capacities up to 24 tokens
per TP lane. With MTP5, seq1–4 use capacities 6/12/18/24 for TP1 and 3/6/9/12
for TP2; allocation and dispatch validation use the actual capacity. TP1 uses
combine stripes 4. TP2 retains the existing `VLLM_ILU_LL_FUSED_AG_AR=1` path
and its fused finalizer; its stripe setting can retain the configured default.
FP8 combine, DeepEP-LL and concurrent/DBO invocation are not supported.

- Decode consumes NVSM fixed-slot INT8 dispatch output directly. The fused
  FC2 stores into the borrowed combine buffer; the ordinary NVSM combine
  wrapper publishes/completes the communication epoch. No input compaction or
  extra collective is added.
- Replay keeps the existing MoE runner communication, expand/quantize and
  weighted reduce. The fused FFN reads compact expert rows and writes local
  BF16 output. An expert with more than 1023 rows is split into contiguous
  groups; each launch has at most 24 groups and 16384 rows. The same expert
  weights can appear in several groups. Shared residual and routing scales
  retain the existing reduce semantics.
- Converted weights cannot fall through to an incompatible legacy GEMM. An
  unsupported path raises an error instead of interpreting the new layout
  as NN weights.

## Validation

The CPU integration tests cover the disabled path, draft exclusion, unsupported
parallelism, registration lifetime and replay chunk coverage. The ixformer
GPU test compares fused replay with a CPU FP32/BF16 reference, and checks the
weight permutation and metadata. The NVSM lifecycle tests cover both 18/24-token
registration and one dispatch → fused FFN → combine sequence.

The fused FFN preserves clamp 10 and dynamic per-row INT8 quantization. Floating
point rounding can differ from separate NN GEMV kernels; component tests do not
replace full-model accuracy evaluation.

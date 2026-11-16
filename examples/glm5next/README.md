# GLM5Next decode reproduction

Base revisions: IxFormer `master` at `6694481724236711efde7285ef7a0dbc3f119f77`; vllm-iluvatar `feat/glm5next-opt` at `fdcae4e55be54f31d385429308b995172cf2da6d`. Build both repositories after applying this change. The independent IXMM Python backend used for the reference is `66a8d0142aa55230146de792de8a2ffd0bbf106e`; it is distinct from IxFormer's submodule pin.

```bash
MODEL=/path/to/GLM5.3-Flash-W4A8-NN-V1 PORT=12354 \
  bash examples/glm5next/run_decode.sh
```

The launch uses TP2/DP8/EP16, budget 2048, MTP5, FULL_DECODE_ONLY, automatic graph sizes and 0.88 memory utilization. The draft is not forced eager. No KV transfer connector or custom worker is needed. KDA state, activations and KV caches use BF16. The existing W8A8 projection, IXMM MoE, shared-expert overlap, gather-before-gate and radix sampling options remain enabled.

## Ported paths

- `VLLM_ILU_GLM5NEXT_MHC_DECODE_SPLITK=1`: keep mHC projection parameters in BF16 during loading; select the validated separate post/pre decode policy for M=5..127. IxFormer chooses the MR split-K=8 kernel from the existing `ds4_mhc_pre_v2` entry point. Its 24-row weights need no padding. The last M/N tile uses overlapping reads and disjoint writes. M<16 has an initialized activation scratch allocation. The existing projection rounding and RMS fusion policy are preserved.
- `cat_output=False` on the mixed KDA projection returns QKV and BF16 gates separately. The consumer splits only the gates, avoiding a concat immediately followed by a split. Gate values and computation precision are unchanged.
- `VLLM_ILU_GLM5NEXT_DRAFT_HEAD_W8A8=1`: quantize a private copy of the loaded MTP vocabulary head. vLLM shares the target head module with MTP, so the module dictionaries must be copied before replacing its weights. The target head remains BF16. The existing argmax and TP reduction path is retained. This is numerically different from a BF16 draft head and must be recorded in accuracy comparisons.
- `VLLM_ILU_SKIP_CUDAGRAPH_MEMORY_PROFILE=1`: skip only the throwaway graph-memory estimate, retaining activation profiling and real FULL decode capture. This preserves the previously validated workaround for the CoreX collective graph lifecycle; it does not patch NCCL or the driver.

The mHC and draft-head switches default off. Disabling the mHC switch retains the branch's sqrsum/allreduce fusion policy. The new MR kernel dispatch itself is based on dtype and token shape, not on a serving-framework environment variable. Unselected experimental bmm, gated norm/quant, KDA launch and target-head quantization paths are not part of this port.

## Focused checks

Run on an available GPU, with the newly built IxFormer and plugin first on the Python path:

```bash
# IxFormer repository
python -m pytest tests/python/inference/functions/models/deepseek_v4/mhc/float/test_mhc_pre.py \
  -k glm5_decode_splitk_boundaries -q

# Plugin repository
python -m pytest tests/models/glm5next/test_projection_quant.py \
  -k 'split_qkv_gates_exact or draft_head_keeps_shared_target_unchanged' -q
```

Model measurement: one formal round at 4 requests per DP (32 total), 112640 input tokens, 600 measured output tokens, 8192-token background requests, pinned DP ranks. Validate all request usages, live background requests, per-DP counters and zero preemptions. The SLA uses original SSE interval P50 divided by 3.2, at most 20 ms; report actual TPOT and whole-round MTP acceptance separately. Do not infer performance from the kernel tests or reuse historical results as this build's measurements.

## 2026-11-16 reproduction on 31.6

IxFormer implementation commit: `a828ebd` on the base above. CoreX compiler: `5.0.0 dfce698b054f757b9ae6c509e42974dfe542da89`, ivcore11; host driver 5.1.0. Both native libraries were rebuilt and their loaded paths checked in all server/worker processes. The existing vLLM source and standalone IXMM backend were retained.

- One valid round, 4 requests per DP / 32 total: SSE P50 **63.53 ms**, SSE P50 / 3.2 **19.853125 ms**, satisfying the 20 ms SLA.
- Actual generated-token TPOT **15.284386 ms**; whole-round MTP acceptance **77.0752%** (202149 accepted / 262275 proposed draft tokens).
- Legacy estimated aggregate throughput **1611.8369 tokens/s**, calculated as 32 × 3.2 × 1000 / 63.53. This is not measured usage/wall-time throughput. Hardware count is 16 logical GPU devices / 8 physical cards.
- All 32 request usages, 8 DP counter populations, background activity and zero preemptions passed. No foreign GPU processes were observed; the server was stopped after the round.
- Focused validation: 13 mHC boundary tests, 7 projection/isolation tests, 22 real-weight/activation cases with repeated graph replay. M=24/36/48 matched the frozen mHC outputs exactly; M=12 had a maximum absolute difference of 0.004855 and passed existing tolerances.

The historical best reference at this concurrency was 19.667188 ms (SSE / 3.2), so this round reproduces the same SLA with a 0.95% higher SSE interval; it does not establish a speedup. The effective cache block changed from 1280 to 2304 on the newer plugin branch (observed hits 97.73% to 96.14%). MTP acceptance changed from 46.65% to 77.08%; counters agree with event counts, but that is not an accuracy result. This timing client does not persist generated token IDs/text, and GSM8K has not been rerun on these revisions. No higher concurrency was scheduled.

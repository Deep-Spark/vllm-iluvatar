# DeepSeek V4.1 producer-prefix state reuse

This optional worker path transfers the current producer's complete prefix-21
hidden state, last 128 SWA rows, compressor rings and source-20 selectors to the
consumer. The consumer validates original NIXL completion, the full current
scheduler allocation and absolute positions of compact remote receive pages.
It keeps the original compressed main/index caches and their publication rules.
A common vote across all EP ranks selects state reuse or the original model.

The initial supported configuration is one host, producer TP2/PP4 with a
21-layer partial model, consumer TP2/DP8/EP16, DSpark K5, and two in-flight
requests per DP. Even prompts with a fully captured current last128 window are
eligible. Partial or missing source capture, local-only cache hits, invalid
receive provenance, changed request generations and unsupported mixed shapes
fall back to the original consumer. No new prefix CUDA Graph is included here.

Use producer worker class
`vllm_iluvatar.v1.worker.prefill_state.host_source_worker.HostSourceWorker`,
consumer worker class
`vllm_iluvatar.v1.worker.prefill_state.live_d_worker.LiveDWorker`, and launch the
consumer through `python -m
vllm_iluvatar.v1.worker.prefill_state.engine_entry serve ...` so scheduler
receipts are installed in spawned engines. Launch the original proxy policy
through `python -m vllm_iluvatar.v1.worker.prefill_state.source_proxy ...`.

All three services must share a fresh `P21_RUN_UUID`, an existing `P21_WIRE`
directory on local tmpfs, `P21_SHADOW_DIR` for producer packets, and
`D21_SHADOW_DIR` for consumer diagnostic receipts. The proxy also accepts
`DSV41_WORKSPACE` to locate the existing `run_script/proxy.py`. These paths are
runtime owners, not persistent caches, and must not be reused across service
lifetimes. Producer contexts are bounded at16 pinned-host slots; each consumer
owns at most two9.5MB packets. Producer GPU buffers and extra communicators
are not allocated. Keep normal target/draft graphs, model length256000, and at
least1024MiB of consumer headroom at the full peak.

Original128/C16 performance:23.241/23.363ms TPOT; returned canonical control:
23.969/24.198ms. Original GPQA10 and a separately routed original question
passed. The feature is explicitly selected and does not change the default
worker or proxy. All original transfer, cancellation and allocator ownership
rules remain in effect.

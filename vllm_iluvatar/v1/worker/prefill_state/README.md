# Complete producer-prefix state reuse (E25)

On DeepSeek V4.1, reuse the producer's complete layer 0–20 state for D's
128/129-token tail replay, then execute layers 21–39 normally. Ordinary KV
transfer, cache publication and receive ownership stay in OptNixl. The original
vLLM scheduler and vllm-router are not modified.

## Standard lifecycle

Use `vllm serve ...` (or `python -m vllm.entrypoints.cli.main serve ...`) with the
normal Iluvatar platform plugin and OptNixl connector. No special worker class,
engine entry point, Python proxy or separate state process is needed.

The supported V4.1 V2-runner PD configuration enables reuse automatically. To
perform a diagnostic comparison, set
`kv_connector_extra_config.enable_prefix_state_reuse=false` in both roles'
`--kv-transfer-config`. This does not disable the ordinary replay alignment or
pipeline KV relays. Full-depth producers and unsupported model/cache layouts
use ordinary inference.

- Platform configuration resolves producer layer cuts and a per-engine session
  identity before spawning workers. The identity is not part of the compilation
  hash and requires no launcher nonce.
- `IluWorker.load_model` asks `IluDeepseekV41ModelState` to create a producer or
  consumer context. The D workspace is allocated **before KV memory profiling**.
- Cache initialization binds existing cache tensors and the native connector.
  Capture/import starts only after warmup and graph capture have completed.
- Standard ModelState request/input/attention callbacks track admissions and
  batches. Worker scheduler-output records track block replacement/extension.
- Normal model forward captures P states; D eager forward uses the optional
  context. Explicit valid-row handling in the decoder layer replaces temporary
  attention/Engram method reassignment. Empty DPs skip prefix attention and
  cache writes but enter the same masked MoE collectives.
- Native connector allocation/completion/build callbacks produce replay receipts
  from successful receives and the scheduler's public current block snapshot.
  No scheduler methods or class methods are replaced at runtime.

## TP/PP and transport

Rank IDs, shard count and boundary ownership derive from actual TP/PP and layer
intervals. The following complete 40-layer cuts retain a nonempty executed
prefix on every producer stage:

| Producer | Full layer partition | Executed prefix counts |
|---|---|---|
| TP2 PP4 | `5,5,5,25` | `5,5,5,6` |
| TP1/TP2 PP8 | `3,3,3,3,3,2,2,21` | `3,3,3,3,3,2,2,2` |
| TP1 PP16 | five `2`, ten `1`, then `20` | five `2`, eleven `1` |

Explicit `VLLM_PP_LAYER_PARTITION` is honored. Without an explicit partition,
configuration distributes the executed prefix across nonempty stages and puts
the unused suffix on the final logical stage. The model, connector and upstream
layer-count helpers see the same partition. Invalid/empty partitions are
reported at configuration time.

Each P engine currently resides on one host. Its local workers export pinned
CPU shards; a thread owned by the existing P connector scheduler assembles them.
D pulls immutable packets through that P's already registered OptNixl side
channel. Multiple P engines use their own identities/endpoints. P and D can run
on separate machines: they need no shared filesystem, shared `/dev/shm`, shared
nonce or extra network port. Addresses must be mutually reachable, and the
normal PD request-ID propagation must be preserved.

Full boundary/cache states are TP-replicated, so a TP1 producer supplies either
D TP rank from its source TP0 packet. TP2 maps corresponding ranks. All local
producer shards must agree on topology, prompt, admission and geometry before
publication. Field ownership, shape, byte layout and checksum are validated.
D also verifies runtime ring capacity and indexer dtype; incompatible geometry
falls back before any GPU write.

State storage defaults to `/dev/shm/vllm_prefix_state`, with a temporary-directory
fallback. `kv_connector_extra_config.prefix_state_root` optionally selects a
host-local root. The configuration-owned engine/session identity isolates
restarts. Completed admissions move to a private publication directory before
joining; a newer admission cannot be deleted by old admission cleanup. Incomplete
shards expire after 3600 seconds; published packets expire after 60 seconds.

## Replay and memory contracts

For prompt length N, the received even prefix is H=N-(N%2), replay starts at
S=H-128, and the query has N-S rows. Wire version 3 captures every row in [S,N).
For 129 rows it also carries the uncached last CR1 main-KV row and indexer
key/scale row. Quantized keys and FP32 scales occupy separate page regions;
restoring them preserves page padding. Compressor rings retain the open CR2
group. The internal boundary field names remain `HC`, `pre_mix`, `topk20` and
`candidates20`.

Compressed main KV uses the same native lifecycle. `PrefixGeometry` carries
`main_kv_format` (`none`, `i8g64`, or `i4g32`) on every packet, including even
prompts. Odd boundary rows carry the existing packed payload and FP16 group
scales as a pair, with identical block/row indices and no requantization. D
binds both cache planes before initializing prefix reuse. The worker leases the
bounded compressed workspace around execution; the complete model forward,
including prefix reuse and the remaining suffix, shares one forward scope.
SWA remains BF16 and indexer storage retains its independent layout.

Before the common EP decision, D checks engine/session/request/prompt/slot
generation, successful original receive, replay shape, current full KV pages,
compact receive-to-logical-page mapping, cache geometry and available memory.
Missing or partial state, failed transfer, preemption, multimodal/embedding/LoRA
identity, adaptive verification, unsupported auxiliary-state layout or a peer
veto takes ordinary forward on all ranks before import.

D copies validated pinned CPU views directly into existing live cache/batch
buffers and retains their owners until a CUDA event confirms completion. No GPU
packet buffer or new graph pool is allocated. The existing dedicated prefix
workspace scales with configured `max_num_seqs * (1 + speculative_tokens)`,
rounded to TP alignment. K5/seq4 uses 983736 logical bytes (~0.938 MiB); allocator
reserve is accounted for separately. There is no seq<=2 or seq<=4 activation
gate; actual decode rows must fit the workspace and the existing LL communicator.
Draft buffers are not reused, and draft graph configuration is independent.

## Tests

CPU tests cover topology ownership, atomic publication, transport, failure
receipts, request reuse, boundary key/scale layout, workspace sizes and explicit
padded-prefix execution. Integration tests use the standard worker/model-state
and connector callbacks. Deployment validation must also check the actual
TP/PP cut, cache geometry, replay hits, memory reserve and end-to-end accuracy.

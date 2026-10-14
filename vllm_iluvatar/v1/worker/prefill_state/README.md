# E25: service-owned complete producer prefix states

E25 reuses the current producer's complete layer0–20 state and executes D's
layer21–39 suffix for a new request's tail128 replay. Ordinary KV transfer,
cache publication and receive ownership remain in OptNixl.

Start both P and D with `python -m vllm_iluvatar.v1.worker.prefill_state.engine_entry
serve ...`. P uses `HostSourceWorker`; D uses `LiveDWorker`. The original
OptNixl-capable vllm-router routes HTTP requests unchanged. No Python proxy or
standalone state process is used. Keep request-ID randomization disabled on
both services. Each service bootstrap generates its own lifetime nonce.

P's existing PP3/TP0 host export thread aggregates all eight local producer
shards. Its scheduler serves versioned metadata and CPU payloads over the
existing registered OptNixl side-channel endpoint. D obtains that endpoint
from its current remote-engine metadata, validates the payload and holds only
pinned host memory. After the common EP vote, it copies fields on the current
stream directly into existing SWA/ring/HC/topk/candidate destinations. Hosts
remain alive until the copy event completes, including request cancellation.

P and D may run on separate hosts. No P/D shared filesystem, shared `/dev/shm`,
shared nonce, new network port or router extension is required. Each P node's
eight TP/PP ranks use that P host's local CPU state directory.
`P21_LOCAL_STATE_ROOT` optionally changes the local storage root (default
`/dev/shm/vllm_p21`, temporary-directory fallback). `P21_SHARD_TIMEOUT_S` and
`P21_PACKET_TTL_S` bound incomplete and published packet retention. The P/D
side-channel addresses must be mutually reachable.

The prefix token budget comes from configured max-num-seqs and speculative
query length. The common vote selects the maximum actual decode count, rounded
to TP alignment. A dedicated small workspace is allocated before KV memory
profiling: HC BF16, pre_mix FP32, positions int64, IDs int32 and padding mask
share one aligned byte owner. K5/seq4 costs983736 logical bytes (~0.938MiB);
allocator reserve is measured separately. No DSpark buffer is reused and no
additional graph pool is captured. Target/draft graph configurations remain
independent. Original runner table/slot views are used without duplicate owners.

Empty DPs skip prefix attention/Engram/cache writes and still join the same
masked MoE collectives. E25 has no configured seq<=2/<=4 activation gate; actual
decode must fit the configured workspace and existing LL communicator capacity.
Adaptive verification's different row layout currently takes ordinary forward.
Missing/partial source, mismatched engine/request/prompt/nonce/generation, stale
allocation/pages, insufficient memory or a peer veto also take ordinary forward
on all EP ranks before any state import.

# E25: service-owned complete producer prefix states

E25 reuses the current producer's complete layer0–20 state and executes D's
layer21–39 suffix for a new request's tail128/129 replay. Ordinary KV transfer,
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

With the connector's two-token receive alignment, a prompt of length N has
received prefix H=N-(N%2), replay start S=H-128 and query length N-S. Even
prompts replay128 rows; odd prompts replay129, including one uncached token.
Wire version2 captures all rows in [S,N). For129 it also carries the last
layer20 CR1 main-KV row and indexer key/scale row. D restores those into the
current live KV slots before suffix execution; the receipt still records H
as the originally received prefix. Quantized indexer keys and FP32 scales
occupy separate page regions and are copied separately, preserving page
padding. CR2 layers2/8/14 keep the unfinished group in their exported ring.
This adds1156 payload bytes for the missing INT8-indexer/main-KV row and
72720 bytes for the additional boundary/SWA row per TP packet. Only P's CPU
pool grows to129 rows. D's device workspace and graph pools do not grow.

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

# Tree-Draft integration map

This document maps the materialized Tree-Draft manifests onto the speculative-decoding paths that already exist in this fork. The goal is to add tree proposals and packed target verification by extending the existing speculative, batching, graph, memory, and sampling machinery. No second inference runtime is needed.

## Existing execution paths

### common/speculative is already the draft-runtime seam

`common/speculative.h` exposes the lifecycle already used by the server:

- `common_speculative_begin()` initializes per-sequence speculative state.
- `common_speculative_process()` feeds target batches back into draft implementations that need target features or synchronized draft state.
- `common_speculative_draft()` produces proposals for selected sequence IDs.
- `common_speculative_accept()` feeds accepted-length statistics back to the selected draft implementation.

The current `common_speculative_draft_params` is linear: one `llama_tokens * result` per sequence. Tree-Draft should evolve this seam to expose a sealed tree view in addition to the existing linear result, rather than introduce a separate request scheduler or model executor.

The existing draft implementations already call `llama_decode(ctx_dft, batch)`, reuse `common_sampler`, and maintain one speculative implementation per server sequence. MTP also attaches backend samplers directly to `ctx_dft`. Tree-Draft sampling and expansion should therefore bind to the same draft context and sampler infrastructure.

### examples/speculative already contains a tree-verification oracle

`examples/speculative/speculative.cpp` is the strongest integration precedent in the repository.

During draft expansion it creates multiple draft branches by:

1. cloning a branch's draft KV state with `llama_memory_seq_cp()`;
2. assigning a distinct `seq_id` to each branch;
3. assigning each shared ancestor token to every branch that descends from it by increasing `batch_tgt.n_seq_id[token]`;
4. assigning branch-local children only to their branch `seq_id`;
5. setting positions by draft depth;
6. calling one `llama_decode(ctx_tgt, batch_tgt)` for all proposed branches.

The target logits are then addressed by the batch indices stored in each branch's `i_batch_tgt`.

This proves that the existing target context, batch allocator, output mapping, memory sequence operations, and decoder can verify a tree-shaped proposal in one target decode without a new target runtime. The limitation is representation: one target KV sequence is consumed per branch. That approach is suitable as a correctness oracle and transitional fallback, but it does not scale to the intended packed tree representation.

### server speculative decoding is linear today

`tools/server/server-context.cpp` currently stores:

- `spec_draft`: a linear token vector;
- `spec_i_batch`: one target batch index for the sampled token plus one index per draft token;
- `spec_ckpt`: checkpoint state used when sequence removal cannot cheaply roll back a partially accepted draft.

`server_slot::handle_last_sampled_token()` appends the sampled token followed by every draft token at consecutive positions. Later, `common_sampler_sample_and_accept_n()` walks those target output rows in order and stops at the first mismatch.

Tree-Draft should preserve this server lifecycle: draft before batch rendering, verify in the normal target decode, accept in `post_decode()`, update speculative feedback, then retain or roll back target/draft memory. The tree-specific work is replacing the linear proposal/index pair with a sealed tree plus packed-node-to-output-row mapping.

A production tree path cannot be represented by `server_batch` unchanged because its token record contains only one `id_slot`, and `render()` calls `common_batch_add(..., { t.id_slot }, ...)`. The tree example relies on one token belonging to multiple branch sequence IDs. The final packed Tree-Draft path should avoid expanding server slots into branch sequence IDs entirely; the existing multi-sequence encoding remains useful as an oracle/fallback.

## Existing llama batching and graph seams

### llama_batch already carries the topology information needed by the oracle path

A `llama_batch` token already has:

- token ID;
- absolute position;
- `n_seq_id`;
- an array of `seq_id` values;
- an output/logits flag.

`llama_batch_allocr::init()` validates shared sequence membership, position monotonicity, coupled sequences, and output selection. `split_equal()` understands sequence sets and coupled sequences. This machinery should remain the outer batch/microbatch mechanism for Tree-Draft.

For the final packed path, Tree-Draft metadata should travel beside the existing `llama_ubatch`, not replace it. The ordinary fields remain authoritative for tokens, positions, request/slot identity, and output selection. Packed parent/depth/tree/KV-indirection metadata is additional graph input.

### llama_context::decode is the target verification entry point

`llama_context::decode()` already:

1. validates and normalizes a `llama_batch` through `llama_batch_allocr`;
2. acquires a memory context;
3. splits work into ubatches;
4. builds and runs the model graph;
5. asynchronously extracts logits and optional backend-sampling outputs;
6. preserves the mapping from user batch token index to output row through `output_ids`.

`llama_get_logits_ith(ctx, batch_index)` resolves that batch index through `output_ids`. A Tree-Draft verifier should keep a packed-node-to-batch-index table so target logits can be associated with tree nodes without assuming output rows are physically contiguous after ubatching.

No new target execution loop is required.

### llama graph attention inputs are the insertion point for true tree masking

Cached attention currently delegates mask construction to the memory context through `llm_graph_input_attn_kv::set_input()` and `mctx->set_input_kq_mask(...)`. No-cache attention explicitly builds visibility from sequence equality plus causal position ordering.

The final Tree-Draft packed verifier needs an additional tree-aware visibility source at this same graph-input layer:

`visible(query_node,key_node) = committed_prefix(key) OR ancestor(key_node,query_node)`.

The existing causal mask remains the default path. Tree metadata should only select the tree-aware mask for a Tree-Draft verification ubatch. This keeps every ordinary decode and current speculative implementation on the existing graph path.

Pillar A's packed topology/mask work should therefore attach to the existing graph-input/memory-context mask construction, rather than create a separate target graph builder.

### KV integration belongs in llama memory, not in common/speculative

The branch-per-`seq_id` oracle duplicates logical sequence state through `llama_memory_seq_cp()`. Pillar B replaces that scaling model with logical tree KV identities and physical indirection.

The correct ownership boundary is:

- `common/speculative`: proposal topology and draft probabilities;
- target batch/ubatch: packed verification tokens and positions;
- target memory context: physical KV placement/lookup for packed tree nodes;
- graph input: ancestry visibility and logical-to-physical KV indirection;
- server: request lifecycle, batching, checkpoint/fallback, result accounting.

Tree-Draft must not own a second KV cache.

## Concrete handoff: sealed draft tree to target packed verification

The handoff should follow these stages.

### 1. Seal the draft proposal

Pillar C `ID_150` yields an immutable topological tree:

`nodes = {token,parent,depth,local_p,path_id,flags}`

with `parent[i] < i`.

For integration, the sealed view also needs the committed prefix length or base position already known by the server slot. The tree does not copy prompt/KV state.

### 2. Canonicalize into the verifier contract

Pillar D `ID_151` should be the bridge contract between the draft tree and target verification.

Map fields directly:

| Draft tree | Target packed verification |
| --- | --- |
| node token | `token_id[]` |
| parent index | `parent[]` |
| depth | `depth[]` |
| local draft probability | `draft_logprob[]` |
| stable path ID | branch/path identity |
| active/pruned flags | active-node mask |
| server prompt next position | committed prefix/base position |

The canonical target order should preserve or deterministically remap the topological order and emit forward/inverse permutations as specified by `ID_152`.

### 3. Build ordinary token/position/output fields

For every packed active node:

- `batch.token[i] = token_id[node]`;
- `batch.pos[i] = prefix_pos + depth[node]`;
- `batch.logits[i] = true` for each node whose target distribution participates in acceptance or residual sampling.

Node-to-batch-index must be stored explicitly. The verifier must not infer target output row from physical row order because `llama_context::decode()` may ubatch and reorder outputs.

The request's normal server slot identity remains attached to the packed forest/request metadata. It is not multiplied into one server slot per branch.

### 4. Materialize tree KV addresses through the existing target memory context

For the correctness fallback, reproduce the example's branch-per-`seq_id` method:

- clone the committed target sequence into temporary branch sequence IDs;
- attach shared ancestors to all descendant branch IDs;
- run one target decode;
- prune temporary branch state after acceptance.

For the production packed path, Pillar B maps each tree node to physical target KV storage while preserving the same `llama_context::decode()` ownership and synchronization model. The verifier supplies logical tree metadata; the memory context supplies KV indices.

### 5. Inject ancestry visibility at the existing attention-mask seam

Pillar A computes packed ancestry. The cached attention graph uses that ancestry in the same place that `set_input_kq_mask()` currently produces causal masks.

For a packed tree query `v`:

- all committed-prefix keys visible to the request remain visible;
- a speculative key `u` is visible iff `u` is an ancestor of `v`, including `u=v`;
- siblings and descendants are masked even when they share the same absolute position;
- another request/tree is always masked.

No model layer should need to understand branches.

### 6. Execute one ordinary target decode

Call the existing target `llama_decode(ctx_tgt, packed_batch)`.

This is the same target execution API already used by both the server and the tree example. Graph construction, scheduler execution, backend selection, output buffers, and synchronization remain owned by `llama_context`.

### 7. Resolve target outputs back to nodes

Use the stored node-to-batch-index map with `llama_get_logits_ith()`, or later a backend verifier path that preserves the same logical mapping.

This mapping is the tree equivalent of `server_slot::spec_i_batch`.

A useful concrete name for the integration object is a packed verification map containing:

- `node_to_batch[]`;
- `batch_to_node[]` for output nodes;
- canonical/original node permutations;
- prefix/base position;
- request/slot ID;
- logical tree/KV descriptor.

It should be a view/descriptor, not an executor.

### 8. Accept a path and commit through existing speculative/server lifecycle

The target verifier returns:

- accepted path/node sequence;
- replacement/bonus target token when required;
- number of speculative tokens accepted;
- selected leaf/path identity.

The server then follows its current responsibilities:

- call `common_speculative_accept()` with acceptance feedback;
- keep the accepted target KV path and discard unaccepted speculative KV;
- update prompt/generated-token accounting;
- restore a checkpoint only for memory backends whose sequence-removal capability requires it;
- continue the same slot/request.

The current `common_sampler_sample_and_accept_n()` remains the linear verifier. Tree acceptance should be added beside it as a tree-aware sampler/verifier function using the same `common_sampler` state and target logits, not as a separate sampler stack.

## Transitional oracle versus production packed path

Two implementations should coexist during bring-up.

The oracle path is the behavior already demonstrated by `examples/speculative`: encode branches as target `seq_id` values, copy prefix KV with `llama_memory_seq_cp()`, and verify all branches in one decode. It is valuable for cross-checking masks, target logits, accepted paths, and rollback behavior.

The production path uses one packed tree description plus tree KV indirection and ancestry masking. It removes the one-sequence-per-branch requirement and is the path intended for server throughput.

Both paths should consume the same sealed draft tree and produce the same logical verifier result. That gives the project a reference implementation without creating a second runtime.

## Independent root manifests that can start first

The current 400-manifest catalog has these dependency roots:

`ID_001, ID_038, ID_044, ID_051, ID_101, ID_151, ID_174, ID_201, ID_251, ID_301, ID_351`.

They are not equally good first implementation targets.

### Start first

**ID_001 - packed tree topology ABI**

This is the highest-value root. It is a data contract and maps directly onto the tree already represented implicitly in `examples/speculative`. It can be implemented as a compact shared descriptor with validators before any custom kernel exists. The oracle path can consume it immediately.

**ID_101 - draft-engine request contract**

Implement this by extending/normalizing the existing `common_params_speculative` / per-sequence speculative configuration boundary. It should remain configuration/state owned by `common_speculative`; it must not introduce another request scheduler.

**ID_151 - target verification input contract**

This is the explicit bridge from the sealed draft tree into the existing `llama_batch` + target-context path. A host-side descriptor and validator can be implemented and tested against the current branch-per-`seq_id` oracle before tree attention or tree KV kernels exist.

**ID_038 - exact top-k reference**

This is safe and independent if implemented as a reference semantic primitive used by Tree-Draft tests and candidate selection. It should align with the current sampler's ordering/tie semantics rather than replace `common_sampler`.

**ID_351 - benchmark scenario schema**

This does not affect runtime architecture and can be implemented immediately. It is useful early because the oracle and packed paths need matched scenarios for correctness/performance comparison.

### Start early, but unify contracts before code duplication

**ID_044 and ID_174 - counter-based RNG contracts**

Both are roots describing schedule-independent RNG, while Pillar C `ID_111` depends on its own path-identity contract. They should converge on one shared counter-addressing primitive/ABI. Do not implement separate RNG engines for sampling and verification.

The implementation order should define the common counter format once, then have draft sampling and target stochastic verification use domain-separated counters.

**ID_051 - logical tree-KV identity**

The contract can start early as a mapping onto existing `llama_memory_i` / KV-cache concepts. Physical page allocation, a second cache object, or server-owned KV storage should wait until the target verifier descriptor is wired to the current memory context.

**ID_201 - SM89 capability contract**

This is independently implementable as a read-only device capability descriptor, but it is not on the functional bring-up critical path. It should not delay the host/oracle integration.

### Defer until the core handoff exists

**ID_251 - model-source abstraction**

The repository already has mature model loading. Starting a new format-neutral loader before Tree-Draft proves a loader-specific need risks creating a parallel loading stack. Treat this root as compatibility work after the draft/target path is functional.

**ID_301 - host request/result envelope**

The server already has tasks, slots, request parsing, streaming, and speculative configuration. A new canonical request runtime at this stage would duplicate server ownership. Bind Tree-Draft into existing server task/slot semantics first; later API work can expose those capabilities without replacing the server runtime.

## Recommended first implementation wave

The first functional wave should be:

1. `ID_001`: packed topology ABI and validation.
2. `ID_101`: Tree-Draft configuration normalized into existing speculative state.
3. `ID_151`: verifier input descriptor plus conversion from the packed topology.
4. `ID_038`: exact reference top-k semantics for deterministic tests.
5. one shared counter-RNG contract serving `ID_044` and `ID_174`, with Pillar C consuming it later.
6. `ID_351`: benchmark scenario schema to compare linear speculation, branch-per-sequence tree oracle, and later packed verification.

With those pieces, a minimal Tree-Draft proposal can be represented, validated, converted to the existing branch-per-`seq_id` target batch, verified through one ordinary `llama_decode()`, and compared against the current speculative implementation. That produces an executable integration skeleton before tree KV pages, Triton masks, CUDA fusion, or new server/API surfaces are required.

## Files and seams to preserve

The integration should keep these existing ownership boundaries:

| Existing file/seam | Tree-Draft use |
| --- | --- |
| `common/speculative.{h,cpp}` | draft implementation selection, draft context ownership, proposal lifecycle, acceptance feedback |
| `common/sampling.{h,cpp}` | sampler state, target acceptance/residual sampling semantics |
| `examples/speculative/speculative.cpp` | branch-per-sequence correctness oracle for tree verification |
| `tools/server/server-context.cpp` | slot lifecycle, target batching, checkpoints, prompt/accounting, speculative stats |
| `src/llama-batch.{h,cpp}` | outer token/position/output normalization and ubatching |
| `src/llama-context.{h,cpp}` | single target decode entry point, output mapping, backend execution |
| `src/llama-graph.{h,cpp}` | tree-aware graph inputs and attention-mask integration |
| `src/llama-memory* / llama-kv-cache*` | physical target KV storage, branch/tree indirection, commit/prune |

The central architectural rule is simple: Tree-Draft produces more structured speculative metadata, but target inference still flows through the existing llama context, graph scheduler, memory system, sampler state, and server slot lifecycle.


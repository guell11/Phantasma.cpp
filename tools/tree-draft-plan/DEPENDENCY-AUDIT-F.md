# Dependency Audit - Pilar F (ID_251..ID_300)

Scope: semantic global-DAG audit for Pilar F against Pilar A quantized/tree attention, Pilar B KV geometry, Pilar C draft execution, Pilar D target verification, and Pilar E device/kernel capability contracts. This file proposes dependency changes only. No manifest is edited.

## Executive finding

Pilar F is internally detailed but currently isolated from the execution contracts it claims to validate. Every dependency in `pillar-f.jsonl` points to another F module, even where the objective explicitly says "tree runtime", "draft and target", "attention/KV geometry", "device capability", or "kernel support".

The missing edges matter because F is not merely a file parser. Its later modules construct runtime model handles, choose quantized kernels and placement, validate draft/target compatibility, and certify attention/KV geometry. Those decisions must consume the already-defined contracts in A-E instead of re-defining compatible-looking local versions.

Use notation `consumer -> dependency`.

## High-confidence cross-pillar edges

| Suggested edge | Why this is a real dependency |
| --- | --- |
| `ID_266 -> ID_236` | F validates storage alignment before materialization. E ID_236 is the global alignment contract combining vector, DMA, tensor-tile, and allocator requirements. Without this edge F can accept an on-disk/buffer alignment that later E kernels reject, or invent a second alignment table. |
| `ID_276 -> ID_022` | F assigns compute and accumulation dtypes across attention/MLP/model roles. A ID_022 owns the tree-attention mixed-precision rules, including fp32 softmax/output accumulation. F's model-wide dtype plan must preserve that attention policy. |
| `ID_286 -> ID_023` | F's device/quant capability matrix includes operation, dtype, layout and fallback eligibility. A ID_023 defines the quantized-KV attention capability boundary and requires unsupported K/V formats to route to an existing backend/staging path. F must expose capabilities consistent with that boundary. |
| `ID_286 -> ID_087` | B ID_087 defines which quantized KV page layouts are legal for Tree PagedAttention. F's capability matrix cannot advertise a quantized KV operation/layout pair that violates B's page-level block/packing constraints. |
| `ID_286 -> ID_201` | E ID_201 is the normalized SM89 device capability descriptor used by Ada-specific choices. F's `C[d,q,op,...]` must consume the real device descriptor rather than duplicate CUDA capability probing. |
| `ID_286 -> ID_216` | E ID_216 defines legal tensor-core dtype candidates for Ada verification/projection GEMMs. F's quant/kernel matrix and mixed-precision planning need the same supported compute modes so loader planning cannot select a dtype the execution policy forbids. |
| `ID_286 -> ID_220` | E ID_220 defines the fused dequantization-to-tensor-core staging contract for quantized weights. F's capability registry must treat this as a concrete kernel path distinct from native packed matmul and generic dequant-on-the-fly fallback. |
| `ID_291 -> ID_106` | F's immutable ModelHandle is the loaded runtime object from which a draft executor is built. C ID_106 defines the draft-model adapter contract (packed input -> logits with exact row mapping). The handle must contain enough validated model/tensor/kernel state to instantiate that adapter without a second loader abstraction. |
| `ID_291 -> ID_164` | The same ModelHandle also backs target execution. D ID_164 defines the one-pass target verification decode boundary and the inputs/results it needs. F should validate that its handle/placement/kernel plan can serve that execution contract before activation. |
| `ID_292 -> ID_166` | F says draft and target must have a "runtime-compatible logits domain". D ID_166 defines the actual target log-probability domain after production sampler transforms. Compatibility is incomplete unless it is stated relative to that target distribution contract. |
| `ID_293 -> ID_153` | F's optional draft->target token-id bijection produces the IDs that target verification must consume. D ID_153 defines target token packing. The remap contract must therefore output exactly the ID domain expected by target packing, before any verification batch is built. |
| `ID_295 -> ID_021` | F validates `H/H_kv` and supported GQA/MQA relations. A ID_021 defines the tree-attention query-head -> KV-head mapping and supported ratios/fallback. F should validate the mapping A will actually execute. |
| `ID_295 -> ID_052` | F explicitly validates attention/KV geometry and exports canonical KV-head/head-dim values. B ID_052 owns fixed-page KV geometry, per-layer K/V strides and alignment. This edge is also independently identified by the Pilar B audit. |
| `ID_295 -> ID_086` | B ID_086 specializes GQA/MQA mapping over paged KV storage. F's loader validation must reject geometries that cannot be represented by the paged-attention head mapping even if the raw architecture dimensions are arithmetically valid. |
| `ID_296 -> ID_088` | F normalizes RoPE/position metadata for runtime use. B ID_088 defines the position semantics for ancestry-shared tree KV. F must normalize to a positional contract that preserves those shared-KV semantics. |

## High-confidence missing edge inside Pilar F

| Suggested edge | Reason |
| --- | --- |
| `ID_275 -> ID_286` | ID_275 dispatches quantized matmul using the predicate `K(device,q,M,N,K,layout)`, but the capability matrix that defines exactly that information is ID_286. Today ID_275 can be implemented only by inventing an earlier ad-hoc capability query. Adding this edge is acyclic: ID_286 depends on ID_271/ID_284, neither of which depends on ID_275. |

The numeric order of IDs should not be treated as a semantic ordering constraint. A later-numbered module can be a dependency when the graph remains acyclic.

## Specification coupling that should not become a reciprocal edge

ID_284 currently says group-size validation also requires `g in supported_group_sizes(device,q)`, while ID_286 is the module that owns device/kernel capability lookup and already depends on ID_284.

Do **not** add `ID_284 -> ID_286`; together with the existing `ID_286 -> ID_284` it would create an artificial cycle. The honest split is:

- ID_284 validates format/tensor group geometry and tail-group legality independent of a device.
- ID_286 combines that valid quant descriptor with device/kernel supported group sizes and reports native-kernel eligibility.

That is a wording/ownership cleanup, not a new dependency edge.

## Medium-confidence edges to review before manifest edits

| Candidate edge | Why it may be useful | Why it may stay transitive |
| --- | --- | --- |
| `ID_299 -> ID_033` | F preflight produces a deterministic fallback plan, while A ID_033 owns the attention backend fallback order. Preflight should not certify a fallback that A will never execute. | If ID_286's capability result already includes A's fallback class via `ID_023`, the preflight can consume it transitively. |
| `ID_299 -> ID_164` | Activation preflight should prove the target verification path is executable. | If `ID_291 -> ID_164` is added, ID_299 already reaches it through its direct dependency on ID_291. |
| `ID_292 -> ID_106` | Draft-target compatibility is checked for models that will be run through C's draft adapter. | With `ID_291 -> ID_106`, ID_292 already depends on the adapter contract through the handles it compares. |
| `ID_296 -> ID_010` | A's composite prefix+tree attention mask depends on exact position interpretation. | B ID_088 is the more concrete shared-KV positional contract; depending on both may over-couple loader metadata to one attention implementation. |

## Edges I do not recommend adding

- Do not add `ID_291 -> ID_150`. The loaded-model handle does not own the sealed draft-tree handoff buffer; C's scheduler/runtime owns that request-local object.
- Do not add `ID_291 -> ID_151`. D's verification input is request-local tree data, not model-loader state. The relevant model dependency is the execution boundary `ID_164`.
- Do not add `ID_293 -> ID_150` merely because C hands draft token IDs forward. The remap is specifically a target-domain concern and is more accurately tied to D's target token packing at ID_153.
- Do not add generic dependencies from F loader parsing modules ID_251..ID_270 to E device modules. Format/header/metadata integrity is device-independent; hardware enters only when alignment, kernel capability, placement, or activation planning begins.
- Do not add `ID_289 -> ID_231` just because both discuss memory budgets. F's residency planner places persistent model tensors; E ID_231 is a temporary speculative-decode device allocator. They may share a backend memory query without sharing allocation ownership.

## Removal audit

There are currently no cross-pillar dependencies in Pilar F to remove. The problem is omission, not an excess of artificial A-E edges.

Within F, I would also keep the apparently redundant direct dependencies of ID_299 on ID_286, ID_288 and ID_291. ID_299 is an aggregator that explicitly executes capability, fallback and handle checks; direct edges document those checks even though some are transitively reachable through other F modules.

## Proposed manifest delta

High-confidence additions only:

```text
ID_266 -> ID_236
ID_275 -> ID_286
ID_276 -> ID_022
ID_286 -> ID_023
ID_286 -> ID_087
ID_286 -> ID_201
ID_286 -> ID_216
ID_286 -> ID_220
ID_291 -> ID_106
ID_291 -> ID_164
ID_292 -> ID_166
ID_293 -> ID_153
ID_295 -> ID_021
ID_295 -> ID_052
ID_295 -> ID_086
ID_296 -> ID_088
```

No removal is recommended in this pass.

## DAG effect

All proposed cross-pillar edges point from F (251..300) to earlier pillars A-E, so they cannot by themselves create a path from an earlier pillar back into F. The only later-numbered dependency proposed inside F is `ID_275 -> ID_286`; inspection of F's current edges shows no path `ID_286 => ID_275`, so that addition is also acyclic.

After manifest edits, the global validator should still explicitly recompute SCCs/topological order rather than relying on numeric ID order.


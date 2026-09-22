# Dependency Audit - Pilar E (ID_201..ID_250)

Scope: semantic audit of Pilar E against Pilar A kernel contracts, Pilar B tree-KV semantics, Pilar C CUDA draft scheduling, Pilar D target verification, Pilar F quantization/model contracts, and Pilar H benchmark/performance gates. This document proposes dependency changes only. No pillar manifest or core source was edited.

## Executive finding

Pilar E currently has no cross-pillar dependencies, but several E modules explicitly optimize data or execution stages whose semantics are owned elsewhere. The missing edges are concentrated in four places: paged-KV metadata, tree-node layout, global CUDA scheduling/capture, and quantized/tuning validation.

The conservative correction set is nine additive edges. I found no high-confidence artificial edge to remove from E. Several current intra-E edges are transitively redundant, but they still represent direct interface consumption and should remain.

Use notation `consumer -> dependency`.

## High-confidence missing edges

| Suggested edge | Rationale |
| --- | --- |
| `ID_213 -> ID_056` | E ID_213 lays out paged-KV metadata for locality. B ID_056 owns the semantic page-table entry `(page_id,generation,base_pos,valid_len)`, lookup rules, and stale-generation rejection. Hardware layout may optimize those fields, but it cannot redefine them. |
| `ID_215 -> ID_081` | E ID_215 reorders verifier KV reads by page/span and restores output order. B ID_081 defines the exact ordered visible-page segment set for each tree query. The locality planner must preserve that semantic gather set rather than derive a second visibility representation. |
| `ID_214 -> ID_102` | E ID_214 chooses a warp-coherent device layout for parent, depth, token, score/log-probability, and flags. C ID_102 owns the compact draft-node record and the invariants for those fields. The SM89 layout should be a physical representation of C's node contract. |
| `ID_221 -> ID_093` | E ID_221 defines global CUDA stream roles for draft, verify, H2D, and D2H work. B ID_093 owns the RESERVE -> WRITE -> PUBLISH -> ATTEND -> VERIFY -> COMMIT/PRUNE ordering and reader-quiescence rules that any overlap plan must preserve. |
| `ID_221 -> ID_141` | C ID_141 defines logical stream/event ownership inside draft expansion. E ID_221 is the broader device stream topology, so it should map or compose C's draft roles instead of inventing an incompatible second ownership scheme. |
| `ID_221 -> ID_164` | D ID_164 defines the single-pass target verification decode boundary. E's `verify` stream role and allowed overlap cannot be specified honestly until the verifier execution unit it schedules is defined. |
| `ID_239 -> ID_145` | C ID_145 owns CUDA Graph capture eligibility for repeated draft-expansion shapes. E ID_239 checks capture eligibility for the complete speculative decode DAG, so full-DAG eligibility must include the draft subgraph predicate and its invalidation rules. |
| `ID_220 -> ID_273` | E ID_220 fuses dequantization directly into tensor-core staging. F ID_273 owns the common dequantization semantics and numerical conformance contract for registered quant encodings. The fused path must implement that contract exactly before any scheduling or performance decision is valid. |
| `ID_250 -> ID_391` | E ID_250 accepts or rejects Ada tuning from speedup, memory, latency, error, and confidence. H ID_391 owns statistically robust deltas against compatible baselines and the confidence/effect-size logic. E should consume those generic comparison results instead of defining a second statistical comparator. |

## High-confidence removals

None.

There are no existing cross-pillar E edges to remove. I also do not recommend deleting intra-E edges merely because another dependency reaches the same ancestor. Examples include `ID_218 -> ID_216`, `ID_224 -> ID_221`, `ID_240 -> ID_222`, `ID_243 -> ID_204`, `ID_244 -> ID_205`, and `ID_250 -> ID_243`: each module directly consumes the named contract even though part of that contract is also reachable transitively.

## Pilar A kernel review

I do not recommend a new direct E -> A edge in the conservative set.

The strongest apparent overlaps are not prerequisite relationships:

- A ID_030 owns the Triton tree-attention autotune key space, while E ID_243 is a broader SM89 occupancy/launch tuner that may apply to CUDA gather, verifier, transfer, or other kernels. Making E depend on ID_030 would incorrectly make the hardware tuner Triton-attention-specific.
- A ID_048/ID_049 own kernel microbenchmark and regression-gate contracts for Pilar A. E ID_250 spans allocator, PCIe, L2, graph, and launch policies, so A's kernel-only benchmark contract is too narrow as E's global acceptance prerequisite. H ID_391 is the correct generic statistical seam.
- E ID_214 reaches A's packed topology contract through the already proposed C audit edge `ID_102 -> ID_001`. Adding `ID_214 -> ID_001` as well would duplicate the topology dependency without adding a new interface boundary.

The intended ownership chain is therefore kernel semantics in A, draft semantic records in C, and SM89 physical layout/tuning in E.

## Pilar B KV review

Three E modules are direct B consumers:

1. `ID_213 -> ID_056` for page-table field semantics and stale-generation safety.
2. `ID_215 -> ID_081` for the exact visible-page gather set.
3. `ID_221 -> ID_093` for safe overlap with KV publication, attention readers, verification, and commit/prune.

I do not recommend `ID_213 -> ID_082` in addition to `ID_213 -> ID_056`. ID_082 is one GPU packing of per-query page segments; ID_213 is specifically about cache-local layout of page descriptors. ID_056 is the semantic descriptor contract E must preserve, while ID_215 already consumes the higher-level gather representation through ID_081.

I also do not recommend coupling E's generic device allocators (`ID_231` through `ID_238`) to B's logical KV page allocator. They manage different ownership domains and can remain independently implementable.

## Pilar C CUDA scheduling review

The clean direction is from E's broader hardware orchestration into C's logical draft contracts:

- `ID_214 -> ID_102`: physical node layout consumes the draft node ABI.
- `ID_221 -> ID_141`: global streams compose the draft expansion stream/event subgraph.
- `ID_239 -> ID_145`: full decode capture eligibility composes the draft expansion capture predicate.

Do not add the reverse edges `ID_141 -> ID_221` or `ID_145 -> ID_239`. C should remain able to define draft semantics and logical scheduling without requiring Ada-specific global orchestration. Reversing either seam would also make a later composition edge prone to a direct C <-> E cycle.

## Pilar D verifier review

`ID_221 -> ID_164` is the only high-confidence direct verifier dependency needed in E.

I do not recommend `ID_215 -> ID_165`. D ID_165 gathers drafted-token target logits, while E ID_215 groups KV/page reads for locality. They are both called "gather" operations but consume different data and different ordering contracts. The relevant semantic source for E ID_215 is B ID_081.

Verifier priority ID_223 does not need an additional direct edge to D ID_164 because it already consumes the stream-role contract from ID_221, which should own the verifier-stage mapping.

## Pilar F quant/model review

The high-confidence E prerequisite is `ID_220 -> ID_273`, not `ID_220 -> ID_286`.

F ID_273 defines what dequantization means numerically for every registered encoding. E ID_220 must preserve that math while changing where dequantization happens. F ID_286 is a capability matrix populated from registered kernels and can classify the fused path later; requiring the capability matrix before defining the fused kernel contract would invert ownership.

The concurrently materialized Pilar F audit proposes `ID_286 -> ID_220`, which is compatible with `ID_220 -> ID_273`: the capability matrix consumes the concrete fused-kernel contract, while that fused kernel consumes F's lower-level dequantization semantics. The chain is `ID_286 -> ID_220 -> ID_273`; F ID_273 has no dependency path back to ID_286, so this composition is acyclic.

Likewise, E ID_216/ID_219 can expose hardware/library tensor-core candidates without depending on the full model preflight ID_299. F's loaded-model/preflight layer can intersect model capabilities with E's runtime hardware candidates during integration.

## Pilar H performance-gate review

`ID_250 -> ID_391` is the narrowest honest dependency.

H ID_391 supplies baseline-compatible statistical deltas and confidence. E ID_250 remains the owner of Ada-specific acceptance thresholds and fallback policy. I do not recommend `ID_250 -> ID_393`: H ID_393 is a CI gate over general Tree-Draft metrics, while E ID_250 is a hardware-policy acceptance matrix that must also be usable outside CI.

Similarly, H should not require E ID_250 as a global prerequisite, because Pilar H must benchmark and gate non-SM89 systems too. Ada-specific evidence can be included conditionally without making the portable benchmark layer depend on an Ada tuning module.

## Cycle analysis

Seven of the nine proposed E edges point to lower-numbered pillars or modules whose current dependency closures do not reference E:

- `ID_213 -> ID_056`
- `ID_215 -> ID_081`
- `ID_214 -> ID_102`
- `ID_221 -> ID_093`
- `ID_221 -> ID_141`
- `ID_221 -> ID_164`
- `ID_239 -> ID_145`

Two edges intentionally point forward because the later pillars own generic contracts E consumes:

- `ID_220 -> ID_273`: F owns dequantization semantics shared by loaders and kernels.
- `ID_250 -> ID_391`: H owns generic statistical baseline comparison.

Both forward dependency closures currently stay outside Pilar E, so neither creates a return path.

I first simulated the current 400-module manifests plus the conservative high-confidence changes from the existing Pilar B and Pilar C dependency audits and the nine E edges above. That graph had 823 edges, topologically visited all 400 modules, and reported no cycle.

The Pilar F audit materialized after that first simulation. Merging all of its proposed high-confidence edges with the B/C/E sets makes the global proposal cyclic, but the cycles are not caused by any E edge. They come from F runtime-ownership edges that point back into C/D while C already points into F preflight/model compatibility:

- `ID_291 -> ID_106 -> ID_299 -> ID_291`
- `ID_291 -> ID_164 -> ID_160 -> ID_153 -> ID_152 -> ID_151 -> ID_150 -> ID_292 -> ID_291`
- `ID_292 -> ID_166 -> ID_164 -> ID_160 -> ID_153 -> ID_152 -> ID_151 -> ID_150 -> ID_292`

With those three F audit edges (`ID_291 -> ID_106`, `ID_291 -> ID_164`, and `ID_292 -> ID_166`) omitted, the combined B/C/F/E proposal topologically visits all 400 modules again. This supports keeping the E directions above and resolving the F/C/D ownership inversion in the global consolidation pass rather than weakening E's hardware dependencies.

The main future E-specific cycle hazards are ownership inversions. In particular, do not later add `ID_145 -> ID_239` after `ID_239 -> ID_145`, and do not make a generic H baseline-comparison module depend on E ID_250 after `ID_250 -> ID_391` is accepted.

## Recommended conservative change set

Add these nine edges:

1. `ID_213 -> ID_056`
2. `ID_215 -> ID_081`
3. `ID_214 -> ID_102`
4. `ID_221 -> ID_093`
5. `ID_221 -> ID_141`
6. `ID_221 -> ID_164`
7. `ID_239 -> ID_145`
8. `ID_220 -> ID_273`
9. `ID_250 -> ID_391`

Remove no edges in the conservative pass. After the manifests are updated by the owning integration pass, rerun the global DAG validator and keep the producer/consumer directions above fixed when later F/H audits are merged.

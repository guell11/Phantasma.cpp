# Pilar C - Draft Model Engine and Fast Sampling / Tree Expansion

Scope: ID_101 through ID_150.

This pillar specifies the lightweight draft runtime that builds a bounded speculative token tree before target-model verification. It covers request/state contracts, deterministic sampling, branch scoring, probabilistic pruning, batched node expansion, multi-request scheduling, CUDA launch policy, acceptance-oriented adaptation, telemetry, and the immutable handoff to the target verifier.

## Design invariants

- The tree is append-only while EXPANDING and immutable after SEALED.
- Every non-root node has parent_i < i, so arena order is a valid topological order.
- Node and frontier capacity are admitted up front. Overflow returns an explicit status at a commit boundary.
- RNG is counter-based and keyed by request seed plus stable path identity, so sampling does not depend on GPU warp order, frontier compaction, or request batching order.
- Draft and verifier RNG use one shared counter-addressing ABI with domain-separated counters; Pilar C does not define a second RNG engine.
- Model probabilities are stored separately from heuristic branch scores. Pruning/ranking never rewrites draft probabilities.
- Multi-child sampling records the exact ordered conditional proposal probability for every emitted sibling so downstream stochastic verification can reconstruct the proposal process.
- Expansion is transactional: model execution, sampling, child reservation, writes, pruning, then commit. A pre-commit failure leaves the previous committed tree readable.
- CUDA overlap is expressed with stream/event ownership and data dependencies. Global device synchronization is not part of the draft-step contract.
- Acceptance-oriented policies may alter width and depth, but hard request budgets and deterministic tie-breaking remain authoritative.
- The sealed handoff exposes a verifier-ready topological view without transferring allocator ownership.
- The sealed tree carries base position, request identity, exact proposal probabilities, and sampler/RNG provenance; it does not copy prompt state or own a second KV cache.

## Module map

| IDs | Area | Output |
| --- | --- | --- |
| ID_101..ID_105 | Core request/tree contracts | Immutable request config, node record, arena, path IDs, lifecycle |
| ID_106..ID_110 | Draft runtime adapter | Model adapter, packed frontier, workspaces, cancellation, status model |
| ID_111..ID_116 | Sampling semantics | Counter RNG, stable softmax, top-k/top-p, categorical and multi-child sampling |
| ID_117..ID_120 | Branch scoring | Cumulative log-probability, length/diversity scoring, total ranking key |
| ID_121..ID_125 | Pruning and width allocation | Beam pruning, probability thresholding, child budgets, entropy/confidence width |
| ID_126..ID_130 | Acceptance-oriented utility | Acceptance proxy/calibration, expected prefix gain, cost-adjusted utility |
| ID_131..ID_135 | Scheduling | Request budgets, action records, fairness, batch formation, stop rules |
| ID_136..ID_140 | Batched tree expansion | Two-phase append, metadata/score writes, next frontier, step transaction |
| ID_141..ID_145 | CUDA launch strategy | Stream/event contract, launch shapes, fusion boundary, scan, graph capture |
| ID_146..ID_150 | Adaptive policy and verifier handoff | Width/depth controllers, composed policy, telemetry, sealed tree view |

## Dependency shape

The local Pilar C DAG is intentionally layered. Root contracts start at ID_101 and ID_102. Sampling depends on stable path identity, scoring depends on exact sampled proposal probabilities, pruning depends on scoring, scheduling depends on pruning plus cost utility, append/commit depends on scheduling and sampling, CUDA policy depends on the logical step contract, and adaptive policy depends on measured verifier feedback interfaces.

No module depends on a later Pilar C ID. Cross-pillar execution contracts are represented as abstract interfaces here so the global 400-module DAG can bind them after all eight pillars are available.

## Detailed module responsibilities

- ID_101: draft request configuration and admission limits.
- ID_102: compact tree-node record.
- ID_103: contiguous arena and packed frontier storage.
- ID_104: stable path identities.
- ID_105: request/tree lifecycle state machine.
- ID_106: abstract draft-model execution adapter.
- ID_107: packed frontier descriptors for batched forwards.
- ID_108: reusable logits/candidate workspaces.
- ID_109: cancellation and early-stop propagation.
- ID_110: backend-neutral status/error taxonomy.
- ID_111: stateless counter-based RNG.
- ID_112: temperature scaling and stable normalization.
- ID_113: deterministic top-k filtering.
- ID_114: top-p filtering and renormalization.
- ID_115: deterministic categorical sampling.
- ID_116: multi-child sampling without replacement.
- ID_117: cumulative path log-probability.
- ID_118: length-normalized branch score.
- ID_119: sibling diversity penalty.
- ID_120: canonical total branch-ranking key.
- ID_121: hard frontier beam pruning.
- ID_122: relative probability-mass pruning.
- ID_123: per-parent integer child-budget allocation.
- ID_124: entropy-adaptive desired width.
- ID_125: confidence-gap width narrowing/widening.
- ID_126: acceptance-oriented utility proxy.
- ID_127: empirical acceptance calibration.
- ID_128: calibrated acceptance branch score.
- ID_129: expected accepted-prefix gain.
- ID_130: compute/memory cost-adjusted utility.
- ID_131: request budget accounting and reservations.
- ID_132: scheduler expansion-action record.
- ID_133: weighted fairness across requests.
- ID_134: deterministic draft-forward batch formation.
- ID_135: expansion stop-reason function.
- ID_136: two-phase child append transaction.
- ID_137: parent-to-child metadata propagation.
- ID_138: probability and score materialization.
- ID_139: next-frontier construction.
- ID_140: logical expansion-step orchestration.
- ID_141: CUDA stream/event ownership.
- ID_142: row-wise sampling launch-shape policy.
- ID_143: fused post-logit pipeline boundary.
- ID_144: device exclusive scan for child reservations.
- ID_145: CUDA Graph capture eligibility/cache key.
- ID_146: acceptance-driven width controller.
- ID_147: accepted-yield depth controller.
- ID_148: composed acceptance-oriented expansion policy.
- ID_149: allocation-free per-step telemetry schema.
- ID_150: immutable verifier handoff contract.

## Implementation boundary

These manifests are Phase 1/2 specifications only. Implementation must bind them to the existing `common/speculative` lifecycle, draft `llama_context`/`llama_decode` execution, `common_sampler` state, outer llama batch/ubatch machinery, and the sealed-tree-to-target-verifier handoff described by `INTEGRATION.md`. The pillar must not create a second request scheduler, draft model executor, sampler stack, or KV cache. Each implementation worker should receive one manifest after the global dependency DAG is checked and make the smallest compatible change at the existing seam.

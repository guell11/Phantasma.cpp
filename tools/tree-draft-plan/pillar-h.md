# Pilar H - Benchmarking, Automated Tests & CI/CD Performance Tracking

IDs: ID_351..ID_400

This pillar defines the measurement, correctness, statistical validation, regression detection, and CI evidence needed to make Tree-Draft performance claims reproducible. It is a Phase 1/2 specification only; it does not implement runtime features.

## Scope map

- ID_351..ID_358: benchmark scenario, timing, corpus, warmup, throughput, TTFT, latency, and phase accounting.
- ID_359..ID_365: tree acceptance, verification efficiency, speedup, host/device memory, KV overhead, and scratch high-water marks.
- ID_366..ID_378: deterministic seed propagation, greedy/logit equivalence, sampled-distribution tests, acceptance semantics, rollback/commit invariants, and benchmark helper unit tests.
- ID_379..ID_387: end-to-end, batch, long-context, stop-condition, fuzz, ASan/UBSan, TSan, and accelerator memory-check coverage.
- ID_388..ID_400: result artifacts, environment fingerprints, baselines, statistical comparisons, CI gates, retention, regression attribution, bisect support, reports, scheduling tiers, and release-readiness evidence.

## Measurement principles

Every performance comparison is tied to a canonical scenario and an environment fingerprint. Primary claims retain the raw counts or samples needed to recompute them. Baseline and Tree-Draft runs use matched prompt, output policy, batch, backend, device, and build configuration whenever the metric requires a direct comparison.

Latency measurements use a monotonic clock and explicit phase boundaries. GPU timing must include the synchronization needed to make phase durations meaningful while keeping observer overhead measurable. Throughput, TTFT, inter-token latency, acceptance metrics, and memory peaks are separate metrics; no aggregate score hides regressions in one dimension.

## Correctness and statistics

Greedy decoding requires exact emitted-token equivalence to the non-speculative target path. Committed-position logits use declared numerical tolerances appropriate to the backend and precision. Sampled decoding is validated statistically with categorical goodness-of-fit and pairwise distribution tests instead of requiring identical random token streams.

Acceptance, rejection, commit, and KV rollback tests use small hand-computable trees plus generated property cases. Batch and concurrent tests verify request isolation. Sanitizer and fuzz jobs focus on bounded high-value surfaces so they are practical in CI.

## CI and regression tracking

Machine-readable result artifacts include scenario hash, revision, build configuration, hardware/software fingerprint, metrics, and sufficient raw evidence. Performance baselines are keyed by compatible environments. Gates combine practical effect thresholds with repeated-sample uncertainty so ordinary benchmark noise does not become a false regression.

Failure bundles retain the exact scenario, result artifact, environment fingerprint, logs, and comparator output. Attribution first detects scenario or environment drift, then narrows code revisions without claiming unsupported causality. CI scheduling separates fast checks, nightly coverage, and dedicated hardware jobs.

## Dependency policy

Dependencies in pillar-h.jsonl are IDs only and intentionally minimal. Pilar H uses internal dependencies where one benchmark or validation component consumes another component's output. Runtime functionality under test is treated as the system-under-test interface rather than guessing cross-pillar implementation IDs before the global 400-module dependency audit.

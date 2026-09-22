# phantasma.cpp development release notes — 2026-09-21

This snapshot is a performance-focused development build of `phantasma.cpp`, derived from `llama.cpp`.

## Ready in this snapshot

- Expert-granular MoE tiering with hardware-aware auto-fit.
- Warm-start expert sidecars.
- Explicit `draft-mtp-tree` speculative runtime.
- Tree verifier with target branch sequence support.
- Greedy verifier fast path using direct argmax when sampling constraints are neutral.
- Incremental private-branch KV synchronization instead of full-prefix cloning.
- Cost-aware Tree-MTP branch handling.
- Flash Attention compatible Tree-MTP path.
- Q4 KV cache presets for VRAM-constrained long-context operation.
- Embedded phantasma.cpp branded web UI.
- Windows one-click launchers for the tested fast and 100K presets.

## Tested Windows presets

### FAST

Run `RUN_PHANTASMA_FAST.bat`.

Key flags:

```text
--spec-type draft-mtp-tree
--spec-draft-n-max 7
--spec-draft-p-split 1.0
-c 16384
--temp 0
--kv-unified
--flash-attn on
-ctk q4_0
-ctv q4_0
--spec-draft-type-k q4_0
--spec-draft-type-v q4_0
-b 256
-ub 64
-fit on
```

RTX 4060 8 GB / Gemma 4 26B A4B Q4_K_M observed high-acceptance warm runs: 65.45, 66.25 and 66.28 tok/s. A lower-acceptance run in the same series measured 48.45 tok/s.

### 100K context lab

Run `RUN_PHANTASMA_100K.bat`.

The 102,400-token context starts successfully on the tested 8 GB GPU with Q4 KV. Auto-fit retained 30 hot expert slots per layer versus 31 in the earlier 16K/F16-KV configuration.

With a short occupied context, warm generation measured 49.61 and 48.65 tok/s. With approximately 60,017 prompt tokens actually occupying the context, generation measured 15.76 tok/s and prompt evaluation measured 208.31 tok/s. This distinction is intentional: allocated context capacity is not the same thing as full-context generation throughput.

## UI

The embedded Svelte UI now uses phantasma.cpp branding, the violet/cyan visual system, a ghost mark, a redesigned empty-chat greeting, branded sidebar treatment and an upgraded composer surface. `npm run check` reports zero errors; the production UI build completes successfully.

## Known limitations

- The current Tree-MTP runtime is optimized for greedy sampling (`temperature <= 0`).
- Extra target-side sibling branches are expensive on the tested RTX 4060 and are disabled in the maximum-throughput preset via `--spec-draft-p-split 1.0`.
- Long occupied contexts remain attention/KV bound even though Q4 KV makes 100K allocation practical.
- Throughput varies with MTP acceptance and workload.
- This working tree was materialized without `.git`; initialize a new local repository before publishing.

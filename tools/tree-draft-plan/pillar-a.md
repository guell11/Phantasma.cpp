# Pilar A - Triton Kernels and Custom Attention Masking

Pilar A decomposes the GPU kernel surface for tree-based speculative decoding into exactly 50 atomic modules. The design is anchored to current fork seams: `ggml/src/ggml-cuda/fattn*` already selects FlashAttention kernels by architecture, masks, GQA ratio, alignment, and KV type; `ggml/src/ggml-cuda/top-k.cu` owns CUDA top-k; `common/sampling.cpp` owns sampler-chain semantics; and `examples/speculative/speculative.cpp` currently expresses draft trees through sequence IDs and batched target verification.

The decomposition also reflects public techniques from Triton fused FlashAttention-v2 style kernels, SpecInfer tree-parallel verification, FlashAttention-3 asynchronous Hopper scheduling, and fused GPU top-k/top-p sampling patterns. These are design references only; the plan does not require an external runtime.

## Module map

| ID | Objective | Dependencies |
|---|---|---|
| ID_001 | Define packed tree topology ABI | - |
| ID_002 | Specify ancestor closure construction | ID_001 |
| ID_003 | Specify dense tree mask reference | ID_001, ID_002 |
| ID_004 | Specify packed forest offsets | ID_001 |
| ID_005 | Specify topology validator | ID_001, ID_004 |
| ID_006 | Specify tree position mapping | ID_001, ID_004 |
| ID_007 | Specify packed QKV gather layout | ID_004, ID_006 |
| ID_008 | Specify tile-level tree visibility predicate | ID_002, ID_004 |
| ID_009 | Specify branch KV indirection | ID_001, ID_007 |
| ID_010 | Specify composite prefix plus tree mask | ID_002, ID_006 |
| ID_011 | Specify ragged attention batch descriptor | ID_004, ID_010 |
| ID_012 | Specify sparse block schedule | ID_008, ID_010, ID_011 |
| ID_013 | Specify sparse schedule compaction | ID_012 |
| ID_014 | Specify compact ancestor bitset format | ID_002 |
| ID_015 | Specify bitset mask decode primitive | ID_014 |
| ID_016 | Define exact tree attention oracle | ID_010 |
| ID_017 | Specify online softmax recurrence | ID_016 |
| ID_018 | Specify baseline Triton tree-attention forward | ID_007, ID_011, ID_015, ID_017 |
| ID_019 | Specify sparse-skipping Triton attention | ID_012, ID_017, ID_018 |
| ID_020 | Specify fused prefix and tree phase attention | ID_010, ID_017, ID_019 |
| ID_021 | Specify GQA/MQA head mapping | ID_018 |
| ID_022 | Specify mixed-precision policy | ID_017, ID_018 |
| ID_023 | Specify quantized-KV compatibility boundary | ID_009 |
| ID_024 | Specify split-KV partial attention | ID_017, ID_019 |
| ID_025 | Specify split-KV reduction | ID_024 |
| ID_026 | Specify FlashAttention-v2 style work partition | ID_020, ID_021 |
| ID_027 | Specify Hopper warp-specialized schedule | ID_026 |
| ID_028 | Specify TMA tensor-descriptor interface | ID_027 |
| ID_029 | Specify guarded FP8 attention path | ID_022, ID_027 |
| ID_030 | Specify Triton autotune key space | ID_026 |
| ID_031 | Specify AOT kernel manifest | ID_030 |
| ID_032 | Specify runtime kernel dispatcher | ID_011, ID_021, ID_030, ID_031 |
| ID_033 | Specify backend fallback matrix | ID_003, ID_018, ID_032 |
| ID_034 | Specify attention correctness oracle suite | ID_003, ID_016, ID_018 |
| ID_035 | Specify numerical tolerance matrix | ID_022, ID_029, ID_034 |
| ID_036 | Specify pathological mask cases | ID_005, ID_034 |
| ID_037 | Specify deterministic attention mode | ID_025, ID_032 |
| ID_038 | Define exact top-k reference | - |
| ID_039 | Specify small-k Triton top-k kernel | ID_038 |
| ID_040 | Specify hierarchical large-vocab top-k | ID_038, ID_039 |
| ID_041 | Specify variable-k batched top-k | ID_039, ID_040 |
| ID_042 | Specify fused logits transform plus top-k | ID_039 |
| ID_043 | Specify top-k categorical sampling | ID_039 |
| ID_044 | Specify counter-based RNG contract | - |
| ID_045 | Specify fused joint top-k top-p sampling | ID_038, ID_043, ID_044 |
| ID_046 | Specify tree branch expansion top-k | ID_001, ID_041 |
| ID_047 | Specify verification-logit gather | ID_004 |
| ID_048 | Specify microbenchmark harness | ID_018, ID_025, ID_039, ID_045 |
| ID_049 | Specify performance regression gates | ID_048 |
| ID_050 | Specify Pilar A integration acceptance contract | ID_032, ID_033, ID_034, ID_035, ID_036, ID_037, ID_045, ID_047, ID_049 |

## Design constraints

- Packed forests use parent-before-child order and ragged offsets.
- A proposal node sees the committed prefix and only its own ancestor chain.
- Online softmax retains fp32 max, normalization sum, and output accumulators.
- Sparse tile skipping is legal only when skipped tiles are provably invisible.
- GQA/MQA, quantized KV, architecture checks, and fallback selection are explicit contracts.
- Top-k and sampling define exact tie-break and RNG semantics before optimization.
- Autotune and AOT contracts are separated from mathematical operator semantics.
- Correctness, pathological topology, numerical tolerance, determinism, and performance gates are independent modules.

## Repository seams observed

- `ggml/src/ggml-cuda/fattn.cu`: vector/tile/MMA FlashAttention selection with architecture, mask, GQA, alignment, and KV-type checks.
- `ggml/src/ggml-cuda/fattn-common.cuh`: attention scale, mask, softmax, and backend conventions.
- `ggml/src/ggml-cuda/top-k.cu`: existing CUDA top-k semantic/performance seam.
- `common/sampling.cpp`: host sampler-chain behavior that fused GPU sampling must preserve.
- `examples/speculative/speculative.cpp`: current multi-sequence tree drafting and batched target verification flow.

## Public technique notes

- Triton fused-attention tutorial provides a FlashAttention-v2 style tiled baseline with online softmax.
- SpecInfer establishes parallel verification over a token tree with tree-structured visibility.
- FlashAttention-3 motivates optional warp-specialized asynchronous schedules on Hopper-class GPUs.
- FlashInfer sampling documents fused GPU top-k/top-p, per-row parameters, deterministic options, and rejection-sampling techniques.

## Research references

- Triton fused attention tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- SpecInfer: https://arxiv.org/abs/2305.09781
- FlashAttention-3: https://arxiv.org/abs/2407.08608
- FlashInfer sampling API: https://docs.flashinfer.ai/api/sampling.html
- FlashInfer top-k API: https://docs.flashinfer.ai/api/topk.html

## Research references

- Triton fused attention tutorial: https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html
- SpecInfer: https://arxiv.org/abs/2305.09781
- FlashAttention-3: https://arxiv.org/abs/2407.08608
- FlashInfer sampling API: https://docs.flashinfer.ai/api/sampling.html
- FlashInfer top-k API: https://docs.flashinfer.ai/api/topk.html

The JSONL file is normative. Every line contains exactly `module_id`, `objective`, `mathematical_spec`, `dependencies`, and `implementation_prompt`.

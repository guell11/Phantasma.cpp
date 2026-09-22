# Pilar F - Model Loaders, Quantization & Format Compatibility

This pillar defines IDs `ID_251` through `ID_300`. Its scope is model ingestion and compatibility planning for the tree-draft runtime. It specifies how model files become validated, immutable runtime handles without implementing speculative decoding itself.

The 50 modules are intentionally atomic. A module may be implemented once the IDs in its `dependencies` list are complete. Dependencies stay inside the pillar where possible and point only to IDs, so orchestration can schedule independent work in parallel.

## Module map

| Range | Area | Main outputs |
| --- | --- | --- |
| ID_251..ID_270 | File formats and storage | Format probe, GGUF/Safetensors parsing, metadata normalization, tensor directories, mmap/buffer storage views, integrity checks, loader errors and fallback rules |
| ID_271..ID_288 | Quantization compatibility | Quant registry, GGML block handling, reference dequantization, quantized matmul planning, GPTQ/AWQ/BnB adapters, group/act-order normalization, kernel capability matrix, repack and dequant fallback |
| ID_289..ID_300 | Runtime contracts and validation | Tensor/layer residency, immutable model handle, draft-target compatibility, vocab remap, optional tree assets, KV/RoPE/MoE/tied-weight checks, preflight plan, conformance matrix |

## Design boundaries

The loader is split into four layers of responsibility:

1. **Source decoding** reads container metadata and tensor descriptors without eagerly decoding tensor payloads.
2. **Canonicalization** converts format- and architecture-specific metadata, names, shapes, layouts, and quant state into stable runtime descriptions.
3. **Execution planning** chooses native quant kernels, repacking, bounded dequantization, mmap/buffer storage, and CPU/accelerator placement using explicit capability and resource constraints.
4. **Activation validation** builds an immutable model handle only after structural, quantization, placement, and draft-target compatibility checks have produced a feasible preflight plan.

Corrupt inputs are terminal errors. Fallback is reserved for unsupported features, resource constraints, or policy choices that have a declared recovery path. This prevents a secondary loader or dequant path from hiding malformed model data.

## Compatibility targets

The manifests cover:

- GGUF single-file and split-shard layouts.
- Safetensors single-file and indexed shard layouts.
- Dense and GGML/GGUF block-quantized tensor encodings.
- GPTQ packed weights, zeros, scales, and activation-order metadata.
- AWQ packed weights, zero points, scales, and packing metadata.
- Serializable bitsandbytes 8-bit, NF4, and FP4 state.
- Native quantized matmul, dequantization on the fly, repacking, and bounded full-tensor dequantization fallback.
- Architecture-sensitive tensor naming, shapes, layouts, MoE experts, tied embeddings, KV geometry, RoPE metadata, and optional tree-draft auxiliary tensors.
- Separate draft and target model loading, exact tokenizer/vocabulary compatibility checks, and provable token-ID remapping.

## Implementation rules carried by the manifests

Every module has exactly five fields:

`module_id`, `objective`, `mathematical_spec`, `dependencies`, and `implementation_prompt`.

The mathematical specification is the acceptance contract. Implementations may use existing llama.cpp infrastructure, but they must preserve the stated invariants and structured error behavior. Loader parsing should stay payload-lazy wherever the module says so, and all offset/size arithmetic must be bounds-checked before creating storage views.

Quantized formats are represented as lazy logical views first. Full floating-point materialization appears only as an explicit, budgeted fallback. Capability lookup must be deterministic for a frozen device/kernel registry, and unknown combinations are unsupported until registered.

The final activation gate is `ID_299`. `ID_300` defines the conformance matrix that exercises parsers, adapters, numerical reference paths, corruption handling, fallback decisions, and the end-to-end preflight contract.


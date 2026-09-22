# Pilar E - Low-Level Memory & Hardware Acceleration for RTX 40/Ada

Scope: IDs ID_201 through ID_250. This pillar specifies SM89-oriented memory, launch, tensor-core, transfer, allocator, graph-capture, affinity, and telemetry work for tree-draft speculative decoding. It is specification-only: each JSONL entry is intended to be independently implementable once its listed dependencies are satisfied.

The design treats Ada tuning as runtime-gated policy. Board-specific cache sizes, PCIe behavior, supported tensor-core modes, NUMA topology, and CUDA library capabilities are discovered or benchmarked instead of assumed. Every aggressive fast path keeps an explicit portable or untuned fallback.

## Module index

| ID | Objective | Dependencies |
| --- | --- | --- |
| ID_201 | Define the SM89 hardware capability contract used by all Ada-specific tuning paths. | - |
| ID_202 | Specify SM89 thread-block geometry candidates for latency-sensitive decode kernels. | ID_201 |
| ID_203 | Model warp scheduling pressure for tree-draft kernels on Ada. | ID_201 |
| ID_204 | Compute theoretical occupancy from registers, shared memory, warps, and block limits. | ID_201 |
| ID_205 | Define register-pressure guardrails for decode and gather kernels. | ID_204 |
| ID_206 | Specify shared-memory carveout policy for kernels that trade L1 capacity for staging space. | ID_201 |
| ID_207 | Budget dynamic shared memory for reusable tree-draft staging buffers. | ID_206 |
| ID_208 | Define global-memory alignment contracts for vectorized loads and stores. | ID_201 |
| ID_209 | Specify coalesced global-memory access rules for warp-level token and KV traffic. | ID_208 |
| ID_210 | Specify asynchronous global-to-shared staging on SM89. | ID_207, ID_209 |
| ID_211 | Model L2 working-set residency for decode-time hot data. | ID_201 |
| ID_212 | Specify CUDA L2 access-policy-window candidates for persistent hot regions. | ID_211 |
| ID_213 | Lay out paged KV cache metadata to improve L2 locality during speculative verification. | ID_211 |
| ID_214 | Lay out tree-node metadata for warp-coherent traversal. | ID_209, ID_211 |
| ID_215 | Specify verifier gather ordering that converts irregular accepted-node reads into locality-friendly batches. | ID_213, ID_214 |
| ID_216 | Define tensor-core datatype policy for Ada verification and projection GEMMs. | ID_201 |
| ID_217 | Specify tensor-core-friendly matrix shape padding and alignment. | ID_216 |
| ID_218 | Specify cuBLASLt algorithm selection for small-batch speculative GEMMs on SM89. | ID_216, ID_217 |
| ID_219 | Specify FP8 tensor-core eligibility for Ada when the runtime stack exposes supported FP8 modes. | ID_216 |
| ID_220 | Specify fused dequantization-to-tensor-core staging for quantized weights. | ID_210, ID_216 |
| ID_221 | Define CUDA stream roles for speculative decode overlap. | ID_201 |
| ID_222 | Define event-based cross-stream dependency rules without host synchronization. | ID_221 |
| ID_223 | Specify CUDA stream priority use for latency-critical verifier work. | ID_221 |
| ID_224 | Model copy/compute overlap efficiency. | ID_221, ID_222 |
| ID_225 | Define pinned-host-memory allocation policy for latency-sensitive transfers. | ID_201 |
| ID_226 | Design a reusable pinned-memory pool to avoid per-step pin/unpin overhead. | ID_225 |
| ID_227 | Specify mapped pinned-memory eligibility for tiny host-produced control data. | ID_225 |
| ID_228 | Tune PCIe DMA transfer chunk sizes for host-device metadata traffic. | ID_225, ID_221 |
| ID_229 | Batch small asynchronous DMA requests to reduce launch and bus transaction overhead. | ID_228 |
| ID_230 | Specify bidirectional PCIe scheduling for simultaneous H2D metadata and D2H results. | ID_224, ID_228 |
| ID_231 | Define the device allocator interface used by speculative decode temporary buffers. | ID_201 |
| ID_232 | Specify stream-ordered cudaMallocAsync/cudaFreeAsync usage. | ID_231, ID_222 |
| ID_233 | Tune CUDA memory-pool reservation and release thresholds. | ID_232 |
| ID_234 | Design a per-step scratch arena for short-lived decode temporaries. | ID_231 |
| ID_235 | Design slab allocation for fixed-size tree nodes and descriptors. | ID_231, ID_214 |
| ID_236 | Centralize alignment requirements across host buffers, device buffers, and tensor tiles. | ID_208, ID_231 |
| ID_237 | Specify stride and pitch rules for batched tree tensors. | ID_236, ID_217 |
| ID_238 | Measure allocator fragmentation and allocation-path latency. | ID_231, ID_233, ID_234, ID_235 |
| ID_239 | Define CUDA Graph capture eligibility for the speculative decode DAG. | ID_221, ID_232 |
| ID_240 | Specify a static CUDA Graph capture path for repeated decode shapes. | ID_239, ID_222 |
| ID_241 | Specify graph parameter updates for changing pointers, scalars, and kernel dimensions. | ID_240 |
| ID_242 | Model CUDA Graph cache capacity and eviction. | ID_240, ID_241 |
| ID_243 | Autotune occupancy-sensitive launch parameters using measured latency. | ID_202, ID_204, ID_205 |
| ID_244 | Specify launch-bounds decisions using measured register and occupancy tradeoffs. | ID_205, ID_243 |
| ID_245 | Discover NUMA topology relevant to the GPU's PCIe attachment. | ID_201 |
| ID_246 | Define host-thread affinity for submission and transfer threads. | ID_245 |
| ID_247 | Align pinned-host allocation locality with the GPU's PCIe/NUMA topology. | ID_225, ID_245, ID_246 |
| ID_248 | Define GPU hardware telemetry needed to validate Ada tuning. | ID_201 |
| ID_249 | Define memory-system telemetry and bottleneck classification. | ID_211, ID_224, ID_238, ID_248 |
| ID_250 | Define the SM89 tuning acceptance matrix and regression thresholds. | ID_243, ID_244, ID_249 |

## Validation contract

The companion `pillar-e.jsonl` must contain exactly 50 JSON objects, one per line, with sequential IDs `ID_201` through `ID_250`. Every object has exactly five fields: `module_id`, `objective`, `mathematical_spec`, `dependencies`, and `implementation_prompt`. Dependencies are module IDs only and intentionally minimal.


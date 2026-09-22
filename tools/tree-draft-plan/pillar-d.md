# Pilar D - Target Model Validation & Single-Pass Verification

Scope: IDs ID_151 through ID_200. This pillar specifies target packing, tree attention and positions, one-pass logits, greedy and stochastic verification, corrected residual sampling, reductions, and correctness fallbacks.

## Coverage

- **ID_151..ID_160**: Input contract, packing, positions, and tree-mask semantics.
- **ID_161..ID_168**: Single-pass target decode and probability extraction.
- **ID_169..ID_172**: Greedy verification.
- **ID_173..ID_185**: Stochastic acceptance, residual sampling, and sampler semantics.
- **ID_186..ID_191**: Scatter, reductions, commit/discard result assembly.
- **ID_192..ID_200**: Correctness fallbacks, differential checks, and conformance.

## Module index

| ID | Objective | Dependencies |
|---|---|---|
| ID_151 | Define target verification input contract | - |
| ID_152 | Canonicalize active tree nodes before packing | ID_151 |
| ID_153 | Pack target token ids into verification order | ID_152 |
| ID_154 | Pack per-node sequence ownership metadata | ID_152 |
| ID_155 | Compute absolute target positions for tree nodes | ID_151, ID_154 |
| ID_156 | Define ancestor visibility relation for tree attention | ID_151 |
| ID_157 | Build dense reference tree attention mask | ID_153, ID_156 |
| ID_158 | Build compact ancestor index lists | ID_152, ID_156 |
| ID_159 | Encode tree mask as packed bitsets | ID_157 |
| ID_160 | Validate target packing invariants | ID_153, ID_154, ID_155, ID_158 |
| ID_161 | Define target single-pass batch layout | ID_153, ID_154, ID_155 |
| ID_162 | Map tree nodes to target logits rows | ID_161 |
| ID_163 | Request only required target logits rows | ID_162 |
| ID_164 | Execute single-pass target verification decode | ID_160, ID_161, ID_163 |
| ID_165 | Gather drafted-token target logits | ID_162, ID_164 |
| ID_166 | Compute stable target log probabilities | ID_164 |
| ID_167 | Validate draft proposal log probabilities | ID_151 |
| ID_168 | Materialize per-node verification probability records | ID_165, ID_166, ID_167 |
| ID_169 | Define greedy tree verification rule | ID_164, ID_162 |
| ID_170 | Compute greedy accepted prefix per branch | ID_169 |
| ID_171 | Select deterministic winning greedy branch | ID_170 |
| ID_172 | Select greedy replacement or bonus token | ID_169, ID_171, ID_163 |
| ID_173 | Define stochastic speculative acceptance probability | ID_168 |
| ID_174 | Generate deterministic verification uniforms | - |
| ID_175 | Evaluate local stochastic accept decisions | ID_173, ID_174 |
| ID_176 | Enforce prefix-stopping stochastic acceptance | ID_175 |
| ID_177 | Define stochastic sibling proposal semantics | ID_151, ID_176 |
| ID_178 | Compute stochastic accepted path | ID_176, ID_177 |
| ID_179 | Define residual distribution after rejection | ID_173 |
| ID_180 | Compute residual probabilities stably | ID_166, ID_179 |
| ID_181 | Sample replacement token from residual distribution | ID_174, ID_180 |
| ID_182 | Fallback when residual mass is numerically zero | ID_180 |
| ID_183 | Support truncated draft proposal distributions | ID_167, ID_179 |
| ID_184 | Support temperature-scaled distributions | ID_166, ID_167, ID_169, ID_173 |
| ID_185 | Handle grammar and token-constraint transforms | ID_166, ID_167 |
| ID_186 | Scatter target row results back to tree nodes | ID_162, ID_165 |
| ID_187 | Reduce accepted depth per branch | ID_170, ID_176, ID_186 |
| ID_188 | Reduce winning branch per sequence | ID_171, ID_178, ID_187 |
| ID_189 | Compact accepted node ids into commit order | ID_188 |
| ID_190 | Compute rejected-node discard set | ID_188, ID_189 |
| ID_191 | Assemble verification result record | ID_172, ID_181, ID_182, ID_189, ID_190 |
| ID_192 | Fallback to linear branch verification on unsupported tree mask | ID_157, ID_191 |
| ID_193 | Fallback to ordinary target decoding on malformed input | ID_160 |
| ID_194 | Fallback on target decode failure | ID_164 |
| ID_195 | Detect non-finite logits and probabilities | ID_163, ID_168 |
| ID_196 | Cross-check tree mask against dense reference | ID_157, ID_158, ID_159 |
| ID_197 | Cross-check single-pass logits against linear replay | ID_162, ID_164 |
| ID_198 | Cross-check stochastic verifier against scalar reference | ID_173, ID_175, ID_180, ID_181, ID_182 |
| ID_199 | Define verification correctness invariants | ID_191, ID_190 |
| ID_200 | Specify end-to-end target verification conformance suite | ID_192, ID_193, ID_194, ID_195, ID_196, ID_197, ID_198, ID_199 |

## Core correctness contract

Verification may commit only one causal root-to-leaf path per sequence. Each target row used for a proposed node must represent the distribution conditioned on the committed prefix plus exactly that node''s accepted ancestors. Siblings and cousins must never see each other. Greedy mode accepts exact target argmax matches. Stochastic mode uses the exact target and proposal distributions after configured sampling transforms, acceptance min(1, p_T / p_D), and corrected residual sampling after rejection. Unsupported masks, malformed inputs, decode failures, non-finite distributions, and sampler-state mismatches use explicit correctness fallbacks.

## Artifact contract

pillar-d.jsonl contains exactly one JSON object per module. Every object has exactly five fields: module_id, objective, mathematical_spec, dependencies, implementation_prompt. Dependencies contain module IDs only. IDs are sequential from ID_151 through ID_200.

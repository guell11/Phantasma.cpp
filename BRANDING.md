# phantasma.cpp brand guide

`phantasma.cpp` is the project name and primary public identity for this repository.

## Positioning

**Tagline:** `Make large local models feel less impossible.`

**Short description:** A high-performance llama.cpp research fork focused on running large sparse MoE models on consumer hardware through expert-granular memory tiering, predictive movement, cache-aware scheduling and speculative decoding work.

The voice should feel technical, fast and slightly irreverent. Prefer concrete performance language over hype. Benchmark claims should always include the hardware, model, quantization and exact configuration used.

## Naming

- Display name: **phantasma.cpp**
- Plain-text name: `phantasma.cpp`
- Repository shorthand: `phantasma`
- Do not capitalize it as `Phantasma.cpp` in the logo or primary wordmark.
- `llama.cpp` remains credited as the upstream project and architectural base.

## Creator

Created and maintained by **guell11**.

- GitHub: <https://github.com/guell11>
- Hugging Face: <https://huggingface.co/guell00>

## Visual system

The ghost mark represents model weights moving between memory tiers while the `.cpp` suffix keeps the project anchored to its native C++ runtime roots.

Primary palette:

| Role | Color | Hex |
| --- | --- | --- |
| Void | Near black | `#090B10` |
| Spectral violet | Violet | `#8B5CF6` |
| CUDA cyan | Cyan | `#22D3EE` |
| Foreground | Off white | `#F8FAFC` |
| Secondary text | Slate | `#94A3B8` |

Primary asset: [`media/phantasma-logo.svg`](media/phantasma-logo.svg).

The logo is intentionally self-contained SVG so GitHub renders it without external fonts, scripts or image hosting. Keep generous empty space around the mark and avoid placing it over noisy backgrounds.

## Repository copy

Preferred one-liner:

> **phantasma.cpp** pushes large local MoE inference further on consumer GPUs by treating VRAM, RAM, transfers and speculative work as one performance problem.

Preferred footer credit:

> Built by **guell11** - GitHub [`@guell11`](https://github.com/guell11) - Hugging Face [`guell00`](https://huggingface.co/guell00).

## Upstream and historical attribution

This repository is derived from `llama.cpp` and contains prior `llama-wackMall` work. Rebranding does not remove or replace upstream copyright, license, NOTICE, contributor or historical attribution. Keep those records intact when publishing the project.

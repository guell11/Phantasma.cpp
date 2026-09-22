<p align="center">
  <img src="media/phantasma-logo.svg" alt="phantasma.cpp" width="760">
</p>

<p align="center">
  <strong>Make large local models feel less impossible.</strong>
</p>

<p align="center">
  High-performance local LLM inference for consumer hardware.<br>
  Expert-granular MoE tiering, predictive memory movement, adaptive caching<br>
  and speculative decoding — engineered as one system.
</p>

<p align="center">
  <a href="https://github.com/guell11"><strong>GitHub @guell11</strong></a>
  &nbsp;&middot;&nbsp;
  <a href="https://huggingface.co/guell00"><strong>Hugging Face guell00</strong></a>
  &nbsp;&middot;&nbsp;
  <a href="LICENSE"><strong>MIT</strong></a>
</p>

---

# phantasma.cpp

**phantasma.cpp** is a high-performance local LLM inference engine focused on one slightly unreasonable goal:

GitHub: [`@guell11`](https://github.com/guell11)
Hugging Face: [`guell00`](https://huggingface.co/guell00)


---

## Demo


https://github.com/user-attachments/assets/18aef630-5e7c-400a-ba49-f88ef87230d9


> The repository also contains the original demo as `phantasma.mp4`.
>
> GitHub does not reliably render repository-local MP4 files through standard Markdown.
> For the best README experience, upload `phantasma.mp4` to a GitHub issue/release/discussion, copy the generated `user-attachments` URL and replace the URL above.

---

## Why phantasma.cpp?

Large local models rarely have a simple compute problem.

They have a **memory movement problem**.

A model may technically run, but performance collapses once inference begins moving large amounts of expert data between CPU memory and GPU memory for every generated token.

Traditional layer-level offloading leaves performance on the table for sparse MoE architectures.

phantasma.cpp attacks the problem at a finer level:

```text
                        phantasma.cpp
                             │
            ┌────────────────┴────────────────┐
            │                                 │
          VRAM                               RAM
            │                                 │
     Hot Experts                        Cold Experts
     Dense Weights                       RAM Pool
       KV Cache                         mmap / pread
            │                                 │
            └──────── Predictive Movement ────┘
                             │
                       MoE Scheduler
                             │
                  Adaptive Expert Cache
                             │
                        Tree-MTP
                             │
                         Inference
```

The goal isn't simply:

> "Fit the model."

The goal is:

> **Keep the expensive data where it is most likely to be needed next.**

---

# Core Architecture

## 1. Expert-granular MoE tiering

Traditional offloading operates primarily at layer granularity.

For sparse MoE models, that can be extremely wasteful.

A single MoE layer may contain dozens or hundreds of expert tensors, while each token activates only a small subset.

phantasma.cpp splits experts individually between compute tiers.

```text
MoE Layer
│
├── Hot Expert  ─────► VRAM
├── Hot Expert  ─────► VRAM
├── Hot Expert  ─────► VRAM
│
├── Cold Expert ─────► RAM
├── Cold Expert ─────► RAM
└── Cold Expert ─────► RAM
```

The currently hottest experts remain GPU-resident.

Cold experts stay in system memory and are accessed only when routing actually selects them.

Instead of asking:

```text
Which layers fit in VRAM?
```

phantasma.cpp asks:

```text
Which experts are worth keeping in VRAM right now?
```

That distinction becomes increasingly important as model size grows.

---

## 2. Hardware-aware auto-fit

At startup, phantasma.cpp evaluates the available memory budget and determines how much of the model can remain GPU-resident.

The engine accounts for:

* dense model weights
* KV cache
* compute buffers
* expert tensors
* speculative decoding memory
* remaining VRAM headroom

It then calculates the available number of hot expert slots.

```text
VRAM
│
├── Dense weights
├── KV cache
├── Compute buffers
├── Speculative state
│
└── Remaining budget
        │
        └── Hot expert slots
```

This removes much of the trial-and-error traditionally involved in manually tuning GPU offload.

> [!IMPORTANT]
> Current auto-fit can still over-budget near maximum context in some configurations.
>
> Set `-c` explicitly when evaluating memory consumption or performance.

---

## 3. Adaptive expert cache

Expert popularity is not static.

Routing behavior changes as generation progresses.

phantasma.cpp tracks router decisions online and maintains a decaying score for each expert.

Default behavior:

```text
Score decay:        0.999
Repin hysteresis:   1.5x
Minimum dwell:      32 tokens
```

When a cold expert becomes sufficiently hotter than a currently pinned expert, the engine can dynamically swap them.

```text
Cold Expert
    │
    │ becomes hot
    ▼
┌───────────────┐
│ Cache Manager │
└───────┬───────┘
        │
        ▼
      VRAM
```

No offline profiling is required.

---

## 4. Warm-start sidecar

The adaptive cache can persist what it learned.

When phantasma.cpp exits, the converged expert heat set is stored in:

```text
<model>.tier
```

On the next launch, that state can be restored immediately.

```text
Cold start
    │
    ▼
Learn routing
    │
    ▼
Converged hot set
    │
    ▼
model.gguf.tier
    │
    ▼
Next launch
    │
    ▼
Warm start
```

Delete the `.tier` file to force a fresh cold start.

No separate offline profiling stage is required.

---

## 5. Demand RAM pool

System RAM can operate as an intermediate cache tier below VRAM.

Enable it with:

```text
LLAMA_EXPERT_RAMPOOL
```

The pool follows the same adaptive scoring and hysteresis system used by the expert cache.

Cold operations can transparently access expert tensors from:

```text
VRAM
  │
  ▼
RAM Pool
  │
  ▼
mmap / storage-backed memory
```

This becomes especially useful when running models substantially larger than available GPU memory.

---

## 6. Predictive expert prefetch

phantasma.cpp can attempt to move expert data **before it is requested**.

The next layer's router can be evaluated early using the current hidden state.

Predicted experts are then prefetched by worker threads while useful compute continues.

```text
Current layer
     │
     ├────► normal compute
     │
     └────► predict next experts
                    │
                    ▼
             Prefetch workers
                    │
                    ▼
             Speculative pool
                    │
                    ▼
                Next layer
```

Enable with:

```text
LLAMA_EXPERT_PREDICT=1
LLAMA_EXPERT_PREFETCH_GB=<size>
```

The objective is to overlap memory movement with computation instead of blocking inference while waiting for transfers.

---

## 7. mmap page hints

When pool-backed expert data no longer needs to remain resident, phantasma.cpp can issue page hints allowing the operating system to reclaim memory.

```text
LLAMA_EXPERT_MADVISE=1
```

In development testing this reduced resident memory by approximately **5.4 GiB** on a 35B configuration.

---

# Tree-MTP

phantasma.cpp includes an experimental **Tree-MTP speculative decoding path**.

Instead of treating speculative decoding as a purely linear sequence of draft tokens, the tree path explores multiple candidate continuations and verifies useful speculative work against the target model.

Conceptually:

```text
                      Token
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
             A1        B1        C1
            /  \      /  \      /  \
          A2   A3   B2   B3   C2   C3
```

The objective is simple:

> Get more accepted target tokens from each expensive verification step.

Performance therefore depends heavily on **draft acceptance**.

High acceptance can produce extremely large throughput gains.

Low acceptance reduces those gains significantly.

That distinction matters when interpreting benchmarks.

---

# Verified Results

Historical development measurements:

Hardware:

```text
GPU:       RTX 3070
VRAM:      8 GB
RAM:       31 GB
Storage:   SSD
Sampling:  greedy / --temp 0
```

Perplexity:

```text
PPL = 1.6088
```

across the tested reference configurations.

| Model                        |          Quant | Stock baseline |                phantasma.cpp | Improvement |
| ---------------------------- | -------------: | -------------: | ---------------------------: | ----------: |
| Qwen3.6-35B-A3B              |  IQ2_M / 11 GB |    42.70 tok/s |                 **74 tok/s** |    **+73%** |
| Qwen3.6-35B-A3B              | Q4_K_M / 20 GB |    26.89 tok/s |              **49.93 tok/s** |    **+86%** |
| Gemma 4 26B A4B              | Q5_K_S / 17 GB |    19.50 tok/s |                 **56 tok/s** |   **+187%** |
| Qwen3.5-122B-A10B            |  IQ2_M / 28 GB |     ~8.0 tok/s |              **10.60 tok/s** |    **+33%** |
| Long context / 67K prompt    |              — |       CUDA OOM | **410.38 tok/s prompt eval** |        runs |
| Qwen3.5-122B / 16 GB RAM cap | IQ3_XS / 34 GB |     2.40 tok/s |               **6.56 tok/s** |  **~+173%** |

> Historical measurements above predate some newer RAM-pool and scheduling work and should not be interpreted as current maximum performance.

---

## 122B under RAM pressure

Qwen3.5 122B IQ2_M (~28 GB) with approximately 16 GB free system RAM:

| Configuration              | Tokens | Throughput |                        Pool hit |
| -------------------------- | -----: | ---------: | ------------------------------: |
| RAMPOOL=10                 |    256 | 1.31 tok/s |                           57.1% |
| RAMPOOL=10                 |   1024 | 2.22 tok/s |                           73.2% |
| RAMPOOL=10 + PREFETCH_GB=2 |    256 | 1.41 tok/s | 57.1% + 21.9% speculative probe |

---

# Tree-MTP — RTX 4060 8 GB

Development measurements from **2026-09-21**.

Model:

```text
Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf
```

Draft/MTP model:

```text
mtp-gemma-4-26B-A4B-it.gguf
```

Configuration:

```text
GPU:       RTX 4060 8 GB
Sampling:  greedy
Slots:     1
```

| Configuration                                    |             Observed throughput |
| ------------------------------------------------ | ------------------------------: |
| Linear MTP, warm                                 |                    ~28.17 tok/s |
| Tree-MTP fast path, F16 KV                       |               peak ~53.62 tok/s |
| Tree-MTP + Q4 KV + n_max=7 + b=256 + ub=64       | **65.45 / 66.25 / 66.28 tok/s** |
| Same preset with lower draft acceptance          |                    ~48.45 tok/s |
| 102,400-token allocation, short occupied context |         **49.61 / 48.65 tok/s** |
| ~60,017 tokens actually occupied                 |      **15.76 tok/s generation** |
| ~60,017 tokens — prompt evaluation               |                **208.31 tok/s** |

### Important

Speculative decoding performance depends heavily on acceptance rate.

The **60–66 tok/s** range is reproducible on high-acceptance runs of the current development workload.

It is **not a guaranteed minimum** for arbitrary prompts.

---

# Windows Quick Start

For the tested RTX 4060 / Gemma 4 configuration, the easiest way to run phantasma.cpp is through the launchers in the repository root.

```text
RUN_PHANTASMA_FAST.bat
RUN_PHANTASMA_100K.bat
BUILD_PHANTASMA_CUDA.bat
```

## FAST

```text
RUN_PHANTASMA_FAST.bat
```

Current throughput-oriented preset:

```text
Tree-MTP
16K context
Q4 KV cache
Flash Attention
n_max = 7
batch = 256
ubatch = 64
1 server slot
```

The launcher:

1. resolves paths relative to the repository;
2. starts the inference server;
3. waits for `/health`;
4. opens the embedded phantasma.cpp UI.

Default address:

```text
http://127.0.0.1:8080
```

---

## 100K context

```text
RUN_PHANTASMA_100K.bat
```

Configured context:

```text
102,400 tokens
```

using Q4 KV cache.

The tested 8 GB configuration starts successfully.

Allocating a large context does not mean the entire context is actively expensive from the first token, however.

As the **occupied attention context** grows, generation performance naturally decreases.

Example:

```text
short occupied context:
~49 tok/s

~60K occupied tokens:
~15.76 tok/s
```

---

# Build from source

Requirements:

```text
CMake 3.18+
C/C++ compiler
CUDA Toolkit for NVIDIA builds
```

CUDA build:

```bash
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-completion
```

On Windows, the repository also includes:

```text
BUILD_PHANTASMA_CUDA.bat
```

---

# Running the server

Basic example:

```bash
./bin/llama-server \
  -m /path/to/model.gguf \
  -c 4096 \
  --port 8080
```

Tiering is enabled by default.

To disable phantasma.cpp MoE tiering:

```text
-no-cmoe
```

For long-context K-quant workloads:

```text
-ctk q8_0
-ctv q8_0
```

Flash Attention and other runtime behavior can be configured normally through the available CLI options.

---

# CLI testing

For non-interactive generation:

```bash
./bin/llama-completion \
  -m /path/to/model.gguf \
  -p "Write a comprehensive technical guide to setting up a home Linux server" \
  -n 1024 \
  -no-cnv \
  -st </dev/null
```

Useful flags:

```text
-n N       number of tokens to generate
-no-cnv    disable conversation mode
-st        single-turn generation
```

For automated testing, piping stdin from `/dev/null` prevents the process from waiting for additional interactive input.

---

# Environment Configuration

| Variable                        | Default | Description                              |
| ------------------------------- | ------: | ---------------------------------------- |
| `LLAMA_EXPERT_S`                |    auto | Hot expert slots per layer               |
| `LLAMA_EXPERT_HOT`              |       — | CSV warm-start seed                      |
| `LLAMA_EXPERT_ADAPT`            |       1 | Adaptive online expert placement         |
| `LLAMA_EXPERT_DECAY`            |   0.999 | Expert score decay                       |
| `LLAMA_EXPERT_TMAX`             |      16 | Maximum tokens for tiered hot path       |
| `LLAMA_EXPERT_RAMPOOL`          |       0 | Demand RAM pool size in GiB              |
| `LLAMA_EXPERT_MADVISE`          |       1 | Enable mmap page hints                   |
| `LLAMA_EXPERT_PREAD`            |       1 | Enable pread staging ring when supported |
| `LLAMA_EXPERT_PREAD_RING_MB`    |    auto | Total pread staging budget               |
| `LLAMA_EXPERT_PREDICT`          |       0 | Predictive expert prefetch               |
| `LLAMA_EXPERT_PREFETCH_GB`      |       0 | Speculative expert pool size             |
| `LLAMA_EXPERT_PREFETCH_THREADS` |       2 | Prefetch worker threads                  |
| `LLAMA_EXPERT_PREFETCH_MB`      |      64 | Maximum in-flight prefetch data          |
| `LLAMA_EXPERT_PREDICT_LOG`      |       — | Prediction trace output                  |
| `LLAMA_EXPERT_STATS`            |       — | Runtime statistics output                |
| `LLAMA_EXPERT_USAGE`            |       — | Expert usage dump                        |
| `LLAMA_EXPERT_TRACE`            |       — | Routed expert trace                      |
| `LLAMA_EXPERT_TRACEX`           |       — | Binary routing + router-input trace      |

---

# Memory hierarchy

The long-term architecture treats local inference memory as a hierarchy rather than a binary CPU/GPU split.

```text
┌──────────────────────────────────────────┐
│                  VRAM                    │
│                                          │
│ Dense weights                            │
│ Hot experts                              │
│ KV cache                                 │
│ Active compute                           │
└───────────────────┬──────────────────────┘
                    │
              promotion / eviction
                    │
┌───────────────────▼──────────────────────┐
│                RAM Pool                  │
│                                          │
│ Warm experts                             │
│ Prefetched experts                       │
│ Staging buffers                          │
└───────────────────┬──────────────────────┘
                    │
               mmap / pread
                    │
┌───────────────────▼──────────────────────┐
│                 Storage                  │
│                                          │
│ Cold model data                          │
│ Future third-tier scheduling             │
└──────────────────────────────────────────┘
```

The scheduler's job is increasingly becoming:

```text
predict
   ↓
place
   ↓
prefetch
   ↓
compute
   ↓
observe
   ↓
adapt
   └──────────────► repeat
```

---

# Correctness

Performance is not useful if inference quality silently breaks.

Development validation has included:

* perplexity comparison
* identical-path output comparison
* greedy decoding tests
* adaptive repin invariant checks
* cold/warm restart testing
* deterministic fresh-server runs
* long-context allocation testing

Historical validation produced perplexity within rounding noise of the reference path under the documented test protocol.

When identical compute paths are forced, output has been verified bit-identical.

Different GPU/CPU execution paths can occasionally produce greedy tie flips due to floating-point rounding, similar to changes caused by batch configuration.

New benchmark claims should preserve:

```text
exact command
exact flags
model
quantization
hardware
context
raw benchmark output
```

---

# Current Status

**ACTIVE DEVELOPMENT**

Current work includes:

* expert-granular MoE placement
* adaptive hot/cold caching
* persistent warm-start sidecars
* RAM-pool tiering
* mmap memory pressure control
* pread staging
* predictive expert prefetch
* Tree-MTP speculative decoding
* long-context memory optimization
* CUDA performance tuning

The engine has been tested with both `qwen35moe` and `gemma4` architecture families.

---

# Current Limitations

Expert-tier state, counters, stores and caches are currently still partially process-global.

For now, use:

```text
one active tiered model/context per process
```

until explicit per-context ownership is completed.

The fused-path graph-build state itself is thread-local.

Auto-fit can also become too aggressive near maximum context sizes.

Explicitly configure:

```text
-c <context>
```

when benchmarking memory limits.

---

# Roadmap

Research currently extends toward:

```text
Expert-granular VRAM scheduling
        │
        ├──► Adaptive cache
        │
        ├──► Predictive prefetch
        │
        ├──► RAM intermediate tier
        │
        ├──► Disk-backed third tier
        │
        ├──► Learned router prediction
        │
        ├──► Semantic expert seeding
        │
        ├──► Markov routing correlation
        │
        ├──► Multi-GPU expert priority
        │
        ├──► Per-layer slot skew
        │
        └──► Speculative Tree-MTP scheduling
```

The broader objective is to make memory placement increasingly **predictive rather than reactive**.

---

# Project Philosophy

Modern local inference is often framed as:

```text
model size
vs.
VRAM size
```

phantasma.cpp approaches it differently.

A sparse model does not need every parameter at the same time.

The important questions are:

```text
What data is needed?

Where should it live?

When will it be needed?

Can we predict that before the compute reaches it?

Can useful work happen while the transfer occurs?
```

Once inference is treated as a scheduling problem, the physical size of the model stops being the only interesting number.

That's the rabbit hole.

**phantasma.cpp lives in it.**

---

# Technical Documentation

For implementation details, architecture notes and research methodology:

[`ARCHITECTURE.md`](ARCHITECTURE.md)

For project identity and visual usage:

[`BRANDING.md`](BRANDING.md)

For licensing and attribution:

[`LICENSE`](LICENSE)
[`NOTICE`](NOTICE)

---

# Prior Art Notice

This repository and [`ARCHITECTURE.md`](ARCHITECTURE.md) publicly document the methods and systems developed for phantasma.cpp.

The first public disclosure of the underlying expert-tiering work is recorded as:

```text
2026-07-26
```

The project preserves required attribution for incorporated upstream and third-party work.

Project-specific additions are distributed under the terms documented in [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

The authors intend this public technical disclosure to document the architecture and its development history.

---

# Author

<p align="center">
  <img src="media/phantasma-logo.svg" alt="phantasma.cpp" width="520">
</p>

<p align="center">
  <strong>phantasma.cpp</strong><br><br>
  Created and maintained by <strong>guell11</strong><br><br>
  <a href="https://github.com/guell11">GitHub @guell11</a>
  &nbsp;&middot;&nbsp;
  <a href="https://huggingface.co/guell00">Hugging Face /guell00</a>
</p>

<p align="center">
  <em>Large models. Small GPUs. Questionable decisions.</em>
</p>

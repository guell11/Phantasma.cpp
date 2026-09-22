# phantasma.cpp Gemma 4 Windows launcher

`scripts/run-gemma4-26b-a4b.bat` is the editable Windows launcher for the
Gemma 4 26B A4B reference setup. It defaults to the CUDA Visual Studio build
in `build-cuda-vs`, the GGUF files already present at the repository root, and
server mode on port 8080.

The launcher defaults to the current Tree-Draft/MTP profile used by this
checkout: context 16384, Q4_0 K/V cache, batch 512, ubatch 128, 10 CPU threads,
Flash Attention, no mmap, one server slot, unified KV, fit enabled, temperature
0, and `draft-mtp-tree` with `n-max=7` and `p-split=1.0`. Main and draft
GPU layers default to `all`. The expert-tiering environment keeps
`LLAMA_EXPERT_S=44`, adaptation enabled, score decay 0.999, and the recorded
fast-profile defaults with RAM pool, speculative prefetch, and async H2D
disabled.

The briefing names an IQ4_XS main model. This checkout currently contains
`Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf`, so that file is
the launcher default. Change `MODEL` near the top of the batch file when using
the IQ4_XS model or another compatible Gemma 4 26B A4B GGUF.

## Build

Configure CUDA for SM89 once if `build-cuda-vs` does not already exist:

```bat
cmake -S . -B build-cuda-vs -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
```

Build the release configurations serially:

```bat
cmake --build build-cuda-vs --config Release --target llama llama-cli llama-server -j 1
cmake --build build-cuda-vs --config RelWithDebInfo --target llama llama-cli llama-server -j 1
```

## Run

Double-click the batch file or run it from a terminal:

```bat
scripts\run-gemma4-26b-a4b.bat
```

Edit `APP=server` to `APP=cli` for interactive CLI use. Edit
`BUILD_CONFIG=Release` to `RelWithDebInfo` when debugging. Extra arguments are
appended to the generated command, so one-off overrides that are safe to repeat
can be supplied after the batch filename.

To verify only the executable and model paths without loading the model:

```bat
scripts\run-gemma4-26b-a4b.bat --check-only
```

This check validates paths and launcher selection only. It does not load either
GGUF and must not be treated as a runtime smoke test or performance result.

To audit the complete Windows delivery without loading either model, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify-gemma4-windows-delivery.ps1
```

The verifier checks the CUDA Visual Studio cache, SM89 configuration, launcher
inputs (including the Tree-Draft/MTP options), both GGUF files, and the
core/CLI/server/CUDA artifacts for Release and RelWithDebInfo. It exits with
code 1 when any required delivery artifact is missing, so an incomplete build
cannot be mistaken for a finished package.

The tier implementation is currently process-global for one active tiered
context/model. Run one tiered Gemma process at a time when using this preset.

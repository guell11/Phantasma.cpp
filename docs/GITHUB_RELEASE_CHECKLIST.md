# GitHub release checklist

## Before the first push

- Confirm `*.gguf`, `build*`, `*.log`, DLL/EXE/object files and local caches are ignored.
- Keep `LICENSE`, `NOTICE`, upstream attribution and contributor history documents intact.
- Verify the README benchmark claims include hardware, model, quantization and exact context/sampling assumptions.
- Build CUDA core, CLI and server in both Release and RelWithDebInfo.
- Run `npm run check` and `npm run build` in `tools/ui`.
- Run `scripts\run-gemma4-26b-a4b.bat --check-only`.
- Run `powershell -ExecutionPolicy Bypass -File scripts\verify-gemma4-windows-delivery.ps1`.
- Smoke-test `RUN_PHANTASMA_FAST.bat` and the embedded UI.
- Do not upload model weights to the Git repository.

## Suggested repository metadata

- Repository: `phantasma.cpp`
- Description: `High-performance llama.cpp research fork for large local MoE models on consumer GPUs.`
- Topics: `llama-cpp`, `local-llm`, `cuda`, `moe`, `speculative-decoding`, `tree-draft`, `mtp`, `inference`

## First push template

```powershell
git init -b main
git add .
git status
git commit -m "<write your own commit message>"
git remote add origin https://github.com/guell11/phantasma.cpp.git
git push -u origin main
```

Run `git status` before the commit and verify that no GGUF/model or build artifact is staged.

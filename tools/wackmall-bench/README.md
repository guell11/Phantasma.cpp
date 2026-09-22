# wackMall benchmark harness

This harness runs each benchmark cell in a fresh `llama-bench` process and records the exact command, selected environment, hardware metadata, raw stdout/stderr and parsed throughput.

It exists to avoid mixing cold-start, warm sidecar and steady-state measurements in one reused context.

## Example

```powershell
py -3 tools/wackmall-bench/run.py `
  --bench build-worker3\bin\Release\llama-bench.exe `
  --model C:\path\model.gguf `
  --pp 16,17,256,512 `
  --tg 128,256 `
  --batch 1024 `
  --ubatch 512 `
  --threads 10 `
  --s 44 `
  --tmax 16,256 `
  --sidecar cold,warm `
  --async-h2d 0 `
  --output artifacts\wackmall-bench
```

Use `--dry-run` to emit the matrix without launching the model.

The harness temporarily preserves an existing `<model>.tier` file around each run. For a cold cell it hides the original sidecar, captures any sidecar created by the process as an artifact, then restores the original. For a warm cell it first backs up the original sidecar, lets the process use and update the sidecar at its normal path, captures the resulting file and restores the original bytes afterwards. Restoration runs even when benchmark execution raises an exception. Runtime knobs that materially change transfer behavior are set explicitly for every child process and recorded in `command.json`; this includes `LLAMA_EXPERT_ASYNC_H2D`, whose benchmark default is 0.

Outputs:

- `hardware.json`: host/GPU metadata.
- `plan.json`: exact expanded matrix.
- `runs/<id>/command.json`: command and environment.
- `runs/<id>/stdout.json`: raw llama-bench JSON when parseable.
- `runs/<id>/stderr.txt`: stderr, including LLAMA_EXPERT_STATS output.
- `results.json` and `results.csv`: normalized cells.

Tree-Draft benchmark scenarios use `scenario.py`. A scenario contains exactly `model`, `draft_model`, `prompt_set`, `batch`, `n_ctx`, `tree_shape`, `sampling`, `seed`, `backend`, and `device`. `canonical_json()` serializes those fields deterministically and `scenario_hash()` returns the SHA-256 identity used to compare scenario inputs. Representative greedy and sampled scenarios live under `fixtures/`.

Tree-Draft prompt corpora use `corpus.py`. A corpus manifest pins the tokenizer by name and SHA-256, then records each prompt in order with its category, relative file, byte SHA-256 and tokenizer-derived token length. `corpus_identity()` hashes the pinned tokenizer plus the ordered `(prompt_sha256, token_length)` pairs. `bind_scenario()` validates the ID_351 scenario, requires its `prompt_set` to equal the corpus ID, and returns both identities. The stdlib fixture corpus `prompt-corpus-smoke-v1.json` covers short, medium, long, code, prose and adversarial tokenization inputs; its test tokenizer treats each UTF-8 byte as one token so lengths can be verified without external packages.

The CSV keeps both requested cell dimensions and normalized llama-bench measurement fields. `batch`, `ubatch` and `threads` are the requested harness values; `bench_n_batch`, `bench_n_ubatch` and `bench_n_threads` are the values reported by llama-bench. The CSV also retains `bench_mode`, `n_prompt`, `n_gen` and `n_depth` from each normalized measurement row.

## Validation

Run the stdlib-only focused tests with:

```powershell
py -3 -m unittest tools/wackmall-bench/test_run.py
py -3 -m unittest tools/wackmall-bench/test_scenario.py
py -3 -m unittest tools/wackmall-bench/test_corpus.py
```

No performance claim should be published from this harness without retaining the raw run artifacts.

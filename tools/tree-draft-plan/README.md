# Tree-Draft execution plan

This directory is the machine-readable execution plan for the Tree-Draft Parallel Speculative Decoding runtime.

The catalog is split into eight pillar files with exactly 50 modules each:

- `pillar-a.jsonl`: ID_001..ID_050
- `pillar-b.jsonl`: ID_051..ID_100
- `pillar-c.jsonl`: ID_101..ID_150
- `pillar-d.jsonl`: ID_151..ID_200
- `pillar-e.jsonl`: ID_201..ID_250
- `pillar-f.jsonl`: ID_251..ID_300
- `pillar-g.jsonl`: ID_301..ID_350
- `pillar-h.jsonl`: ID_351..ID_400

Each non-empty JSONL line must contain exactly:

```json
{
  "module_id": "ID_001",
  "objective": "...",
  "mathematical_spec": "...",
  "dependencies": [],
  "implementation_prompt": "..."
}
```

Run:

```powershell
py -3 tools/tree-draft-plan/validate.py
```

The validator rejects missing/duplicate IDs, extra or missing fields, malformed dependencies, self-dependencies, references to unknown modules and dependency cycles. On success it writes:

- `catalog.jsonl`: canonical ID-sorted 400-module catalog.
- `dag.json`: dependency graph, topological levels, roots, terminal nodes and critical path.
- `DAG.md`: human-readable orchestration summary.

Implementation starts only after the validator passes. A worker receives exactly one module manifest at a time. Its result is integrated only after the module-specific static checks, interface checks and tests pass.

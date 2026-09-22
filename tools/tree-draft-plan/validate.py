#!/usr/bin/env python3

import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FIELDS = {
    "module_id",
    "objective",
    "mathematical_spec",
    "dependencies",
    "implementation_prompt",
}
ID_RE = re.compile(r"^ID_(\d{3})$")
PILLARS = {
    "a": (1, 50),
    "b": (51, 100),
    "c": (101, 150),
    "d": (151, 200),
    "e": (201, 250),
    "f": (251, 300),
    "g": (301, 350),
    "h": (351, 400),
}


def fail(errors):
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    raise SystemExit(1)


def load_pillars():
    errors = []
    modules = {}

    for pillar, (start, end) in PILLARS.items():
        path = ROOT / f"pillar-{pillar}.jsonl"
        if not path.exists():
            errors.append(f"missing {path.name}")
            continue

        entries = []
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_no}: invalid JSON: {exc}")
                continue

            if not isinstance(entry, dict):
                errors.append(f"{path.name}:{line_no}: manifest must be an object")
                continue
            if set(entry) != FIELDS:
                missing = sorted(FIELDS - set(entry))
                extra = sorted(set(entry) - FIELDS)
                errors.append(
                    f"{path.name}:{line_no}: field mismatch missing={missing} extra={extra}"
                )
                continue

            module_id = entry["module_id"]
            match = ID_RE.fullmatch(module_id) if isinstance(module_id, str) else None
            if not match:
                errors.append(f"{path.name}:{line_no}: invalid module_id {module_id!r}")
                continue
            number = int(match.group(1))
            if number < start or number > end:
                errors.append(
                    f"{path.name}:{line_no}: {module_id} outside pillar range "
                    f"ID_{start:03d}..ID_{end:03d}"
                )

            for field, minimum in (
                ("objective", 24),
                ("mathematical_spec", 48),
                ("implementation_prompt", 48),
            ):
                value = entry[field]
                if not isinstance(value, str) or len(value.strip()) < minimum:
                    errors.append(
                        f"{path.name}:{line_no}: {field} must be a substantive string "
                        f"(>= {minimum} characters)"
                    )

            deps = entry["dependencies"]
            if not isinstance(deps, list) or not all(isinstance(dep, str) for dep in deps):
                errors.append(f"{path.name}:{line_no}: dependencies must be a string list")
                deps = []
            if len(deps) != len(set(deps)):
                errors.append(f"{path.name}:{line_no}: duplicate dependency in {module_id}")
            if module_id in deps:
                errors.append(f"{path.name}:{line_no}: self dependency in {module_id}")
            for dep in deps:
                if not ID_RE.fullmatch(dep):
                    errors.append(
                        f"{path.name}:{line_no}: malformed dependency {dep!r} in {module_id}"
                    )

            if module_id in modules:
                errors.append(
                    f"duplicate module_id {module_id}: {modules[module_id]['_source']} "
                    f"and {path.name}:{line_no}"
                )
            else:
                entry["_source"] = f"{path.name}:{line_no}"
                modules[module_id] = entry
                entries.append(module_id)

        expected = [f"ID_{i:03d}" for i in range(start, end + 1)]
        if entries != expected:
            missing = [mid for mid in expected if mid not in entries]
            unexpected = [mid for mid in entries if mid not in expected]
            errors.append(
                f"{path.name}: IDs must be sequential and complete; "
                f"missing={missing} unexpected={unexpected}"
            )
        if len(entries) != 50:
            errors.append(f"{path.name}: expected 50 manifests, got {len(entries)}")

    expected_all = {f"ID_{i:03d}" for i in range(1, 401)}
    actual_all = set(modules)
    if actual_all != expected_all:
        errors.append(
            f"global ID coverage mismatch: missing={sorted(expected_all - actual_all)} "
            f"unexpected={sorted(actual_all - expected_all)}"
        )

    if errors:
        fail(errors)
    return modules


def build_dag(modules):
    errors = []
    dependents = defaultdict(list)
    indegree = {}

    for module_id, entry in modules.items():
        deps = entry["dependencies"]
        indegree[module_id] = len(deps)
        for dep in deps:
            if dep not in modules:
                errors.append(f"{module_id}: dependency {dep} does not exist")
            else:
                dependents[dep].append(module_id)

    if errors:
        fail(errors)

    roots = sorted(mid for mid, degree in indegree.items() if degree == 0)
    queue = deque(roots)
    topo = []
    level = {mid: 0 for mid in roots}

    while queue:
        current = queue.popleft()
        topo.append(current)
        for nxt in sorted(dependents[current]):
            level[nxt] = max(level.get(nxt, 0), level[current] + 1)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(topo) != len(modules):
        cyclic = sorted(mid for mid, degree in indegree.items() if degree > 0)
        fail([f"dependency graph contains a cycle involving: {cyclic}"])

    levels = defaultdict(list)
    for mid in topo:
        levels[level[mid]].append(mid)
    for ids in levels.values():
        ids.sort()

    terminal = sorted(mid for mid in modules if not dependents[mid])

    longest = {}
    parent = {}
    for mid in topo:
        deps = modules[mid]["dependencies"]
        if not deps:
            longest[mid] = 1
            parent[mid] = None
            continue
        best = max(deps, key=lambda dep: longest[dep])
        longest[mid] = longest[best] + 1
        parent[mid] = best

    critical_end = max(topo, key=lambda mid: longest[mid])
    critical_path = []
    cursor = critical_end
    while cursor is not None:
        critical_path.append(cursor)
        cursor = parent[cursor]
    critical_path.reverse()

    return {
        "module_count": len(modules),
        "edge_count": sum(len(entry["dependencies"]) for entry in modules.values()),
        "root_modules": roots,
        "terminal_modules": terminal,
        "levels": {str(k): v for k, v in sorted(levels.items())},
        "topological_order": topo,
        "critical_path": critical_path,
        "critical_path_length": len(critical_path),
    }


def write_outputs(modules, dag):
    catalog = ROOT / "catalog.jsonl"
    with catalog.open("w", encoding="utf-8", newline="\n") as handle:
        for i in range(1, 401):
            entry = dict(modules[f"ID_{i:03d}"])
            entry.pop("_source", None)
            handle.write(json.dumps(entry, ensure_ascii=True, separators=(",", ":")) + "\n")

    (ROOT / "dag.json").write_text(
        json.dumps(dag, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Tree-Draft dependency DAG",
        "",
        f"- Modules: {dag['module_count']}",
        f"- Dependency edges: {dag['edge_count']}",
        f"- Parallel levels: {len(dag['levels'])}",
        f"- Root modules: {len(dag['root_modules'])}",
        f"- Terminal modules: {len(dag['terminal_modules'])}",
        f"- Critical path length: {dag['critical_path_length']}",
        "",
        "## Critical path",
        "",
        " -> ".join(dag["critical_path"]),
        "",
        "## Parallel levels",
        "",
    ]
    for key, ids in dag["levels"].items():
        lines.append(f"- Level {key}: {', '.join(ids)}")
    lines.extend([
        "",
        "## Independent roots",
        "",
        ", ".join(dag["root_modules"]),
        "",
        "## Terminal modules",
        "",
        ", ".join(dag["terminal_modules"]),
        "",
    ])
    (ROOT / "DAG.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    modules = load_pillars()
    dag = build_dag(modules)
    write_outputs(modules, dag)
    print(
        "OK: "
        f"{dag['module_count']} modules, "
        f"{dag['edge_count']} edges, "
        f"{len(dag['levels'])} levels, "
        f"critical path={dag['critical_path_length']}"
    )


if __name__ == "__main__":
    main()

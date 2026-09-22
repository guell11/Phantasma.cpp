#!/usr/bin/env python3

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ID_RE = re.compile(r"^ID_(\d{3})$")
PILLARS = {
    range(1, 51): ROOT / "pillar-a.jsonl",
    range(51, 101): ROOT / "pillar-b.jsonl",
    range(101, 151): ROOT / "pillar-c.jsonl",
    range(151, 201): ROOT / "pillar-d.jsonl",
    range(201, 251): ROOT / "pillar-e.jsonl",
    range(251, 301): ROOT / "pillar-f.jsonl",
    range(301, 351): ROOT / "pillar-g.jsonl",
    range(351, 401): ROOT / "pillar-h.jsonl",
}


def load_all():
    modules = {}
    order = {}
    for path in PILLARS.values():
        entries = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            entry = json.loads(raw)
            modules[entry["module_id"]] = entry
            entries.append(entry["module_id"])
        order[path] = entries
    return modules, order


def normalize_edge(edge):
    if not isinstance(edge, dict) or set(edge) != {"consumer", "dependency", "reason"}:
        raise ValueError("each edge needs exactly consumer, dependency, reason")
    consumer = edge["consumer"]
    dependency = edge["dependency"]
    reason = edge["reason"]
    if not ID_RE.fullmatch(consumer) or not ID_RE.fullmatch(dependency):
        raise ValueError(f"bad edge IDs: {consumer!r} -> {dependency!r}")
    if consumer == dependency:
        raise ValueError(f"self dependency: {consumer}")
    if not isinstance(reason, str) or len(reason.strip()) < 20:
        raise ValueError(f"edge reason too short: {consumer} -> {dependency}")
    return consumer, dependency, reason.strip()


def write_pillars(modules, order):
    for path, ids in order.items():
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for module_id in ids:
                handle.write(
                    json.dumps(
                        modules[module_id],
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )


def assert_acyclic(modules):
    indegree = {}
    dependents = defaultdict(list)
    for module_id, entry in modules.items():
        indegree[module_id] = len(entry["dependencies"])
        for dependency in entry["dependencies"]:
            if dependency not in modules:
                raise SystemExit(f"unknown dependency after reconciliation: {module_id} -> {dependency}")
            dependents[dependency].append(module_id)

    queue = deque(sorted(module_id for module_id, degree in indegree.items() if degree == 0))
    seen = 0
    while queue:
        module_id = queue.popleft()
        seen += 1
        for consumer in sorted(dependents[module_id]):
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                queue.append(consumer)

    if seen != len(modules):
        cyclic = sorted(module_id for module_id, degree in indegree.items() if degree > 0)
        raise SystemExit(f"cross-dependency reconciliation would create a cycle: {cyclic}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply reviewed cross-pillar dependency edges to Tree-Draft manifests"
    )
    parser.add_argument(
        "--edges",
        type=Path,
        default=ROOT / "cross-dependencies.json",
        help="JSON object with add/remove edge arrays",
    )
    parser.add_argument("--write", action="store_true", help="write changes to pillar JSONL files")
    args = parser.parse_args()

    spec = json.loads(args.edges.read_text(encoding="utf-8"))
    if set(spec) != {"add", "remove"}:
        raise SystemExit("edge file must contain exactly add and remove arrays")

    modules, order = load_all()
    audit = []

    for action in ("remove", "add"):
        seen = set()
        for raw_edge in spec[action]:
            consumer, dependency, reason = normalize_edge(raw_edge)
            key = (consumer, dependency)
            if key in seen:
                raise SystemExit(f"duplicate {action} edge: {consumer} -> {dependency}")
            seen.add(key)
            if consumer not in modules or dependency not in modules:
                raise SystemExit(f"unknown edge endpoint: {consumer} -> {dependency}")
            deps = modules[consumer]["dependencies"]
            if action == "add":
                changed = dependency not in deps
                if changed:
                    deps.append(dependency)
            else:
                changed = dependency in deps
                if changed:
                    deps.remove(dependency)
            audit.append(
                {
                    "action": action,
                    "consumer": consumer,
                    "dependency": dependency,
                    "changed": changed,
                    "reason": reason,
                }
            )

    for entry in modules.values():
        entry["dependencies"] = sorted(
            entry["dependencies"],
            key=lambda value: int(ID_RE.fullmatch(value).group(1)),
        )

    assert_acyclic(modules)
    print(json.dumps(audit, indent=2))
    if args.write:
        write_pillars(modules, order)
        print(f"wrote {sum(1 for row in audit if row['changed'])} dependency changes")
    else:
        print("dry-run only; pass --write to modify pillar manifests")


if __name__ == "__main__":
    main()

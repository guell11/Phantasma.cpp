#!/usr/bin/env python3
"""Compile and export one fixed-shape grouped expert Triton kernel artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "configs" / "profiles.json"


def load_profile(name: str) -> dict[str, int]:
    data = json.loads(PROFILES.read_text(encoding="utf-8"))
    for profiles in data.values():
        for profile in profiles:
            if profile["name"] == name:
                return profile
    raise ValueError(f"unknown profile: {name}")


def load_runtime():
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is required for AOT compilation")
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError("Triton is required for AOT compilation")
    try:
        import torch
        import triton  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch and Triton are required for AOT compilation") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    sys.path.insert(0, str(ROOT / "kernels"))
    from grouped_expert import warmup_grouped_expert_f16

    return torch, warmup_grouped_expert_f16


def make_schedule(torch, n_tokens: int, n_experts: int, top_k: int):
    buckets = [[] for _ in range(n_experts)]
    for token in range(n_tokens):
        for route in range(top_k):
            buckets[(token * top_k + route) % n_experts].append(token)
    token_ids = []
    offsets = [0]
    for bucket in buckets:
        token_ids.extend(bucket)
        offsets.append(len(token_ids))
    max_group_size = max((b - a for a, b in zip(offsets, offsets[1:])), default=0)
    return (
        torch.tensor(token_ids, device="cuda", dtype=torch.int32),
        torch.tensor(offsets, device="cuda", dtype=torch.int32),
        max_group_size,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="small_m_m1")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build-aot")
    return parser.parse_args()


def json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "_asdict"):
        return json_safe(value._asdict())
    return repr(value)


def main() -> int:
    args = parse_args()
    if args.top_k <= 0 or args.top_k > args.experts:
        print("error: top-k must be positive and no larger than experts", file=sys.stderr)
        return 2
    try:
        torch, warmup = load_runtime()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        profile = load_profile(args.profile)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    x = torch.empty((args.tokens, args.k), device="cuda", dtype=torch.float16)
    weight = torch.empty((args.experts, args.n, args.k), device="cuda", dtype=torch.float16)
    token_ids, offsets, max_group_size = make_schedule(torch, args.tokens, args.experts, args.top_k)
    out = torch.empty((token_ids.numel(), args.n), device="cuda", dtype=torch.float32)

    try:
        compiled = warmup(x, weight, token_ids, offsets, out, profile, max_group_size)
    except Exception as exc:
        print(f"error: Triton warmup/AOT API failed: {exc}", file=sys.stderr)
        return 2

    asm = getattr(compiled, "asm", None)
    if not isinstance(asm, dict):
        print("error: installed Triton did not expose compiled.asm for artifact export", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for kind in ("cubin", "ptx"):
        blob = asm.get(kind)
        if blob is None:
            continue
        suffix = ".cubin" if kind == "cubin" else ".ptx"
        path = args.output_dir / f"grouped_expert_{args.profile}{suffix}"
        if isinstance(blob, str):
            path.write_text(blob, encoding="utf-8")
        else:
            path.write_bytes(bytes(blob))
        artifacts[kind] = path.name

    if not artifacts:
        print("error: Triton compile completed but exposed neither cubin nor ptx", file=sys.stderr)
        return 2

    manifest = {
        "kernel": "grouped_expert_matmul_f16",
        "supported_dtype": "fp16",
        "profile": profile,
        "shape": {"tokens": args.tokens, "experts": args.experts, "top_k": args.top_k, "k": args.k, "n": args.n},
        "artifacts": artifacts,
        "metadata": json_safe(getattr(compiled, "metadata", None)),
    }
    manifest_path = args.output_dir / f"grouped_expert_{args.profile}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Benchmark grouped expert Triton launch profiles for a synthetic routing shape."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "configs" / "profiles.json"


def dependency_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def load_runtime():
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is required for autotuning")
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError("Triton is required for autotuning")
    try:
        import torch
        import triton
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch and Triton are required for autotuning") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    sys.path.insert(0, str(ROOT / "kernels"))
    from grouped_expert import launch_grouped_expert_f16

    return torch, triton, launch_grouped_expert_f16


def load_profiles(mode: str) -> list[dict[str, int]]:
    data = json.loads(PROFILES.read_text(encoding="utf-8"))
    if mode not in data:
        raise ValueError(f"unknown mode: {mode}")
    return data[mode]


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


def performance_metrics(
    ms: float,
    n_assignments: int,
    n_experts: int,
    max_group_size: int,
    n_cols: int,
    k_dim: int,
    block_m: int,
) -> tuple[float, float]:
    useful_tflops = (2.0 * n_assignments * n_cols * k_dim) / (ms * 1.0e9)
    m_tiles = (max_group_size + block_m - 1) // block_m
    launched_rows = n_experts * m_tiles * block_m
    row_utilization = n_assignments / launched_rows if launched_rows else 1.0
    return useful_tflops, row_utilization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("small_m", "prefill"), default="small_m")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--output", type=Path, help="optional JSON file for the measured winner")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k <= 0 or args.top_k > args.experts:
        return dependency_error("top-k must be positive and no larger than experts")
    try:
        torch, triton, launch = load_runtime()
    except RuntimeError as exc:
        return dependency_error(str(exc))

    profiles = load_profiles(args.mode)
    torch.manual_seed(0)
    x = torch.randn((args.tokens, args.k), device="cuda", dtype=torch.float16)
    weight = torch.randn((args.experts, args.n, args.k), device="cuda", dtype=torch.float16)
    token_ids, offsets, max_group_size = make_schedule(torch, args.tokens, args.experts, args.top_k)
    out = torch.empty((token_ids.numel(), args.n), device="cuda", dtype=torch.float32)

    rows = []
    for profile in profiles:
        launch(x, weight, token_ids, offsets, out, profile, max_group_size)
        torch.cuda.synchronize()
        ms = float(
            triton.testing.do_bench(
                lambda: launch(x, weight, token_ids, offsets, out, profile, max_group_size),
                warmup=args.warmup,
                rep=args.rep,
            )
        )
        useful_tflops, row_utilization = performance_metrics(
            ms,
            token_ids.numel(),
            args.experts,
            max_group_size,
            args.n,
            args.k,
            profile["block_m"],
        )
        rows.append({"profile": profile["name"], "ms": ms, "useful_tflops": useful_tflops, "row_utilization": row_utilization})
        print(f"{profile['name']}: {ms:.4f} ms, {useful_tflops:.3f} TFLOP/s, row_utilization={row_utilization:.3f}")

    winner = min(rows, key=lambda row: row["ms"])
    print(f"winner: {winner['profile']} ({winner['ms']:.4f} ms)")
    if args.output:
        payload = {
            "mode": args.mode,
            "shape": {"tokens": args.tokens, "experts": args.experts, "top_k": args.top_k, "k": args.k, "n": args.n},
            "winner": winner,
            "results": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify grouped expert prefix scheduling and, when available, Triton output."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROFILES = ROOT / "configs" / "profiles.json"


def make_schedule(n_tokens: int, n_experts: int, top_k: int) -> tuple[list[int], list[int]]:
    if n_tokens <= 0 or n_experts <= 0 or top_k <= 0 or top_k > n_experts:
        raise ValueError("tokens/experts/top-k must be positive and top-k must not exceed experts")
    buckets = [[] for _ in range(n_experts)]
    for token in range(n_tokens):
        for route in range(top_k):
            expert = (token * top_k + route) % n_experts
            buckets[expert].append(token)

    token_ids: list[int] = []
    offsets = [0]
    for bucket in buckets:
        token_ids.extend(bucket)
        offsets.append(len(token_ids))
    return token_ids, offsets


def verify_prefix_bounds(token_ids: list[int], offsets: list[int], n_tokens: int, n_experts: int) -> int:
    if len(offsets) != n_experts + 1:
        raise AssertionError("prefix array length mismatch")
    if offsets[0] != 0 or offsets[-1] != len(token_ids):
        raise AssertionError("prefix endpoints do not cover the packed assignment list")
    if any(a > b for a, b in zip(offsets, offsets[1:])):
        raise AssertionError("expert prefix bounds must be monotonic")
    if any(token < 0 or token >= n_tokens for token in token_ids):
        raise AssertionError("packed token id out of range")
    return max((b - a for a, b in zip(offsets, offsets[1:])), default=0)


def verify_weight_tile_layout() -> None:
    n_experts, n_cols, k_dim = 2, 5, 7
    stride_w_e = n_cols * k_dim
    stride_w_n = k_dim
    stride_w_k = 1
    flat = list(range(n_experts * stride_w_e))

    expert = 1
    n0, k0 = 3, 5
    block_n, block_k = 4, 4
    old = []
    for k in range(k0, k0 + block_k):
        old_row = []
        for n in range(n0, n0 + block_n):
            if n < n_cols and k < k_dim:
                index = expert * stride_w_e + n * stride_w_n + k * stride_w_k
                old_row.append(flat[index])
            else:
                old_row.append(0)
        old.append(old_row)

    native_block = []
    for n in range(n0, n0 + block_n):
        native_row = []
        for k in range(k0, k0 + block_k):
            if n < n_cols and k < k_dim:
                index = expert * stride_w_e + n * stride_w_n + k * stride_w_k
                native_row.append(flat[index])
            else:
                native_row.append(0)
        native_block.append(native_row)
    transposed_block = [list(row) for row in zip(*native_block)]

    if old != transposed_block:
        raise AssertionError("native [N,K] block load transpose changed the logical [K,N] weight tile")


def load_profile(name: str) -> dict[str, int]:
    data = json.loads(PROFILES.read_text(encoding="utf-8"))
    for profiles in data.values():
        for profile in profiles:
            if profile["name"] == name:
                return profile
    raise ValueError(f"unknown profile: {name}")


def load_triton():
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is required for CUDA verification")
    if importlib.util.find_spec("triton") is None:
        raise RuntimeError("Triton is required for CUDA verification")
    try:
        import torch
        import triton  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch and Triton are required for CUDA verification") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    sys.path.insert(0, str(ROOT / "kernels"))
    from grouped_expert import launch_grouped_expert_f16

    return torch, launch_grouped_expert_f16


def verify_triton(args: argparse.Namespace, token_ids: list[int], offsets: list[int], max_group_size: int) -> None:
    torch, launch = load_triton()
    torch.manual_seed(0)
    x = torch.randn((args.tokens, args.k), device="cuda", dtype=torch.float16)
    weight = torch.randn((args.experts, args.n, args.k), device="cuda", dtype=torch.float16)
    token_ids_t = torch.tensor(token_ids, device="cuda", dtype=torch.int32)
    offsets_t = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    out = torch.empty((len(token_ids), args.n), device="cuda", dtype=torch.float32)

    profile = load_profile(args.profile)
    launch(x, weight, token_ids_t, offsets_t, out, profile, max_group_size)

    ref = torch.empty_like(out)
    for expert in range(args.experts):
        begin, end = offsets[expert], offsets[expert + 1]
        if begin == end:
            continue
        ids = token_ids_t[begin:end].to(torch.int64)
        ref[begin:end] = x.index_select(0, ids).float() @ weight[expert].float().transpose(0, 1)

    torch.cuda.synchronize()
    max_abs = float((out - ref).abs().max().item()) if out.numel() else 0.0
    if not torch.allclose(out, ref, atol=args.atol, rtol=args.rtol):
        raise AssertionError(f"Triton output mismatch: max_abs={max_abs:.6g}")
    print(f"triton: PASS profile={args.profile} max_abs={max_abs:.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true", help="run only prefix-bound reference checks")
    parser.add_argument("--require-triton", action="store_true", help="fail instead of skipping when Triton/CUDA is unavailable")
    parser.add_argument("--profile", default="small_m_m2")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--atol", type=float, default=0.15)
    parser.add_argument("--rtol", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token_ids, offsets = make_schedule(args.tokens, args.experts, args.top_k)
    max_group_size = verify_prefix_bounds(token_ids, offsets, args.tokens, args.experts)
    print(f"prefix: PASS assignments={len(token_ids)} max_group_size={max_group_size}")
    verify_weight_tile_layout()
    print("weight-layout: PASS")
    if args.cpu_only:
        return 0

    try:
        verify_triton(args, token_ids, offsets, max_group_size)
    except RuntimeError as exc:
        if args.require_triton:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"triton: SKIP ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

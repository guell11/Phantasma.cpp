"""Grouped expert scheduling prototype for the optional Triton research lane.

The kernel intentionally supports FP16 inputs and weights only. Quantized
IQ4_XS/Q4_K production work remains in the validated CUDA backend.
"""

from __future__ import annotations

from typing import Mapping

import torch
import triton
import triton.language as tl


@triton.jit
def grouped_expert_matmul_f16(
    x_ptr,
    weight_ptr,
    token_ids_ptr,
    expert_offsets_ptr,
    out_ptr,
    n_cols,
    stride_x_m,
    stride_x_k,
    stride_w_e,
    stride_w_n,
    stride_w_k,
    stride_out_m,
    stride_out_n,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    start = tl.load(expert_offsets_ptr + expert)
    end = tl.load(expert_offsets_ptr + expert + 1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    assignment = start + offs_m
    valid_m = assignment < end
    token = tl.load(token_ids_ptr + assignment, mask=valid_m, other=0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x = tl.load(
            x_ptr + token[:, None] * stride_x_m + offs_k[None, :] * stride_x_k,
            mask=valid_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        weight_block = tl.make_block_ptr(
            base=weight_ptr + expert * stride_w_e,
            shape=(n_cols, K),
            strides=(stride_w_n, stride_w_k),
            offsets=(pid_n * BLOCK_N, k0),
            block_shape=(BLOCK_N, BLOCK_K),
            order=(1, 0),
        )
        weight = tl.load(weight_block, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(x, tl.trans(weight))

    tl.store(
        out_ptr + assignment[:, None] * stride_out_m + offs_n[None, :] * stride_out_n,
        acc,
        mask=valid_m[:, None] & (offs_n[None, :] < n_cols),
    )


def _check_inputs(x: torch.Tensor, weight: torch.Tensor, token_ids: torch.Tensor, expert_offsets: torch.Tensor) -> None:
    tensors = (x, weight, token_ids, expert_offsets)
    if not all(t.is_cuda for t in tensors):
        raise ValueError("all grouped expert tensors must be CUDA tensors")
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise ValueError("grouped expert Triton prototype supports FP16 x and weight only")
    if token_ids.dtype != torch.int32 or expert_offsets.dtype != torch.int32:
        raise ValueError("token_ids and expert_offsets must use torch.int32")
    if x.ndim != 2 or weight.ndim != 3 or token_ids.ndim != 1 or expert_offsets.ndim != 1:
        raise ValueError("expected x[ T,K ], weight[ E,N,K ], token_ids[ A ], expert_offsets[ E+1 ]")
    if weight.shape[2] != x.shape[1]:
        raise ValueError("x K dimension must match weight K dimension")
    if expert_offsets.numel() != weight.shape[0] + 1:
        raise ValueError("expert_offsets must contain n_experts + 1 prefix entries")
    if not x.is_contiguous() or not weight.is_contiguous() or not token_ids.is_contiguous() or not expert_offsets.is_contiguous():
        raise ValueError("grouped expert prototype currently requires contiguous tensors")


def _meta(config: Mapping[str, int]) -> dict[str, int]:
    required = ("block_m", "block_n", "block_k", "num_warps", "num_stages")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing Triton config keys: {', '.join(missing)}")
    return {
        "BLOCK_M": int(config["block_m"]),
        "BLOCK_N": int(config["block_n"]),
        "BLOCK_K": int(config["block_k"]),
        "num_warps": int(config["num_warps"]),
        "num_stages": int(config["num_stages"]),
    }


def launch_grouped_expert_f16(
    x: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    config: Mapping[str, int],
    max_group_size: int,
) -> None:
    _check_inputs(x, weight, token_ids, expert_offsets)
    if out.dtype != torch.float32 or not out.is_cuda or not out.is_contiguous():
        raise ValueError("out must be a contiguous CUDA torch.float32 tensor")
    if out.shape != (token_ids.numel(), weight.shape[1]):
        raise ValueError("out shape must be [n_assignments, N]")
    if max_group_size <= 0:
        return

    meta = _meta(config)
    grid = (
        weight.shape[0],
        triton.cdiv(max_group_size, meta["BLOCK_M"]),
        triton.cdiv(weight.shape[1], meta["BLOCK_N"]),
    )
    grouped_expert_matmul_f16[grid](
        x,
        weight,
        token_ids,
        expert_offsets,
        out,
        weight.shape[1],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        out.stride(0),
        out.stride(1),
        K=x.shape[1],
        **meta,
    )


def warmup_grouped_expert_f16(
    x: torch.Tensor,
    weight: torch.Tensor,
    token_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    out: torch.Tensor,
    config: Mapping[str, int],
    max_group_size: int,
):
    """Compile a fixed launch without executing it and return Triton's compiled kernel."""
    _check_inputs(x, weight, token_ids, expert_offsets)
    meta = _meta(config)
    grid = (
        weight.shape[0],
        triton.cdiv(max_group_size, meta["BLOCK_M"]),
        triton.cdiv(weight.shape[1], meta["BLOCK_N"]),
    )
    return grouped_expert_matmul_f16.warmup(
        x,
        weight,
        token_ids,
        expert_offsets,
        out,
        weight.shape[1],
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        out.stride(0),
        out.stride(1),
        K=x.shape[1],
        grid=grid,
        **meta,
    )

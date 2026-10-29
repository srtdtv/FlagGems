# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MetaX specialization for BF16 activation times INT8 packed weight.

The small-M path follows the two-step Split-K structure already used by the
MetaX BF16 MM implementation.  INT8 values are converted to BF16 before dot,
FP32 partials are reduced, and the per-output-channel scale is applied only
after that reduction. Large contiguous BF16 shapes use a temporary converted
weight and the existing MetaX BF16 MM, followed by channel scaling. Other
shapes retain the generic FlagGems scale-after-dot implementation.
"""

import torch
import triton
import triton.language as tl

from flag_gems.ops.weight_int8pack_mm import (
    weight_int8pack_mm as _generic_weight_int8pack_mm,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from .mm import mm as _metax_mm


_SMALL_M_VALUES = (1, 16, 32, 64)
_SPLIT_K = 8
_BLOCK_N = 64
_BLOCK_K = 128
_REDUCE_BLOCK = 256


@libentry()
@triton.jit
def weight_int8pack_convert(Q, W, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < SIZE
    value = tl.load(Q + idx, mask=mask, other=0).to(tl.bfloat16)
    tl.store(W + idx, value, mask=mask)


@libentry()
@triton.jit
def weight_int8pack_output_scale(Y, S, O, N: tl.constexpr,
                                SIZE: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < SIZE
    value = tl.load(Y + idx, mask=mask, other=0).to(tl.float32)
    scale = tl.load(S + idx % N, mask=mask, other=0)
    tl.store(O + idx, (value * scale).to(tl.bfloat16), mask=mask)


def _can_use_large_m_separate(a, b, scales):
    return (
        a.shape[0] >= 128
        and b.shape[0] % _BLOCK_N == 0
        and a.shape[1] % (_BLOCK_K * _SPLIT_K) == 0
        and a.dtype == torch.bfloat16
        and b.dtype == torch.int8
        and scales.dtype == torch.float32
        and a.is_contiguous()
        and b.is_contiguous()
        and scales.is_contiguous()
    )


def _large_m_separate(a, b, scales):
    # Keep scale after GEMM. MetaX MM materializes BF16 before this scale,
    # introducing one rounding point versus an FP32 fused epilogue.
    # Neither the converted weights nor the unscaled output persist.
    with torch_device_fn.device(a.device):
        converted = torch.empty(b.shape, device=b.device, dtype=torch.bfloat16)
        weight_int8pack_convert[(triton.cdiv(b.numel(), 1024),)](
            b, converted, b.numel(), BLOCK=1024)
        unscaled = _metax_mm(a, converted.t())
        output = torch.empty_like(unscaled)
        weight_int8pack_output_scale[(triton.cdiv(output.numel(), 1024),)](
            unscaled, scales, output, output.shape[1], output.numel(), BLOCK=1024)
    return output


@libentry()
@triton.jit
def weight_int8pack_mm_splitk_partial(
    A,
    B,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    grid_n = N // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_iters_per_split = K // (BLOCK_K * SPLIT_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(
        pid_k * k_iters_per_split, (pid_k + 1) * k_iters_per_split
    ):
        offsets_k = k_iter * BLOCK_K + tl.arange(0, BLOCK_K)
        a_ptrs = A + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offsets_m[:, None] < M, other=0.0)
        b = tl.load(
            B + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
        )
        acc += tl.dot(
            a,
            b.to(tl.bfloat16),
            out_dtype=tl.float32,
            allow_tf32=False,
        )

    partial_ptrs = (
        P
        + pid_k * M * N
        + offsets_m[:, None] * N
        + offsets_n[None, :]
    )
    if EVEN_M:
        tl.store(partial_ptrs, acc)
    else:
        tl.store(partial_ptrs, acc, mask=offsets_m[:, None] < M)


@libentry()
@triton.jit
def weight_int8pack_mm_splitk_reduce_scale(
    P,
    Scales,
    C,
    M,
    N,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for split_id in range(0, SPLIT_K):
        acc += tl.load(
            P + split_id * M * N + offsets, mask=mask, other=0.0
        )
    scales = tl.load(Scales + offsets % N, mask=mask, other=0.0).to(tl.float32)
    tl.store(C + offsets, (acc * scales).to(tl.bfloat16), mask=mask)


def _can_use_small_m_splitk(a, b, scales):
    if a.ndim != 2 or b.ndim != 2 or scales.ndim != 1:
        return False
    m, k = a.shape
    n = b.shape[0]
    return (
        m in _SMALL_M_VALUES
        and n % _BLOCK_N == 0
        and k % (_BLOCK_K * _SPLIT_K) == 0
        and a.dtype == torch.bfloat16
        and b.dtype == torch.int8
        and a.is_contiguous()
        and b.is_contiguous()
        and scales.is_contiguous()
    )


def _small_m_splitk(a, b, scales):
    m, k = a.shape
    n = b.shape[0]
    block_m = 64 if m == 64 else 16
    partials = torch.empty(
        (_SPLIT_K, m, n), device=a.device, dtype=torch.float32
    )
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    partial_grid = (
        triton.cdiv(m, block_m) * triton.cdiv(n, _BLOCK_N),
        _SPLIT_K,
    )
    reduce_grid = (triton.cdiv(m * n, _REDUCE_BLOCK),)

    with torch_device_fn.device(a.device):
        weight_int8pack_mm_splitk_partial[partial_grid](
            a,
            b,
            partials,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(1),
            b.stride(0),
            SPLIT_K=_SPLIT_K,
            BLOCK_M=block_m,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            EVEN_M=m % block_m == 0,
            num_warps=4,
            num_stages=2,
            pipeline="basic",
        )
        weight_int8pack_mm_splitk_reduce_scale[reduce_grid](
            partials,
            scales,
            output,
            m,
            n,
            SPLIT_K=_SPLIT_K,
            BLOCK_SIZE=_REDUCE_BLOCK,
            num_warps=4,
        )
    return output


def weight_int8pack_mm(a, b, scales):
    """Dispatch measured small/large-M paths, preserving generic fallback."""
    assert a.ndim == 2 and b.ndim == 2, "A and B must be two-dimensional"
    assert scales.ndim == 1, "scales must be one-dimensional"
    assert a.shape[1] == b.shape[1], (
        f"incompatible K dimensions: A.shape[1]={a.shape[1]}, "
        f"B.shape[1]={b.shape[1]}"
    )
    assert b.shape[0] == scales.shape[0], (
        f"incompatible N dimensions: B.shape[0]={b.shape[0]}, "
        f"scales.shape[0]={scales.shape[0]}"
    )
    assert b.dtype == torch.int8, f"B must be int8, got {b.dtype}"
    if _can_use_small_m_splitk(a, b, scales):
        return _small_m_splitk(a, b, scales)
    if _can_use_large_m_separate(a, b, scales):
        return _large_m_separate(a, b, scales)
    return _generic_weight_int8pack_mm(a, b, scales)

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

import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

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

# Explicit candidates are consumed by the offline full-call tuner.  Keeping
# them next to the kernels makes the executable search space auditable and
# avoids encoding a performance policy as an algorithm capability check.
DENSE_CONFIGS = (
    (16, 64, 256, 4, 3),
    (32, 32, 128, 4, 3),
    (32, 64, 128, 4, 3),
    (64, 64, 128, 4, 3),
    (64, 128, 64, 8, 3),
    (32, 128, 64, 2, 3),
    # Official FP16 large-M tile retained as a BF16 migration candidate.
    (64, 128, 64, 8, 3),
    # Current production and historical diagnostic controls.
    (16, 64, 128, 4, 2),
    (32, 64, 128, 4, 2),
    (128, 128, 32, 8, 2),
)

SPLITK_CONFIGS = (
    (16, 64, 128, 4, 2, 4),
    (16, 64, 128, 4, 2, 8),
    (32, 64, 128, 4, 2, 4),
    (32, 64, 128, 4, 2, 8),
    (64, 64, 128, 4, 2, 4),
    (64, 64, 128, 4, 2, 8),
)

M1_DENSE_CONFIGS = ((16, 64, 256, 4, 3),)


def _common_rejection_reason(a, b, scales):
    if a.ndim != 2 or b.ndim != 2 or scales.ndim != 1:
        return "requires A[ M,K ], B[ N,K ], scales[ N ]"
    if min(a.shape) == 0 or min(b.shape) == 0:
        return "requires nonempty matrices"
    if a.shape[1] != b.shape[1] or b.shape[0] != scales.shape[0]:
        return "incompatible M/N/K dimensions"
    if a.dtype != torch.bfloat16 or b.dtype != torch.int8:
        return "requires BF16 activation and INT8 weight"
    if scales.dtype != torch.float32:
        return "requires FP32 per-output-channel scales"
    if a.device != b.device or a.device != scales.device:
        return "all operands must share a device"
    if not (a.is_contiguous() and b.is_contiguous() and scales.is_contiguous()):
        return "requires contiguous operands"
    return None


def dense_rejection_reason(a, b, scales, config, m1=False):
    reason = _common_rejection_reason(a, b, scales)
    if reason is not None:
        return reason
    expected = 5
    if not isinstance(config, (tuple, list)) or len(config) != expected:
        return "dense config must be (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages)"
    if any(not isinstance(value, int) or value <= 0 for value in config):
        return "dense config values must be positive integers"
    if m1 and a.shape[0] != 1:
        return "M=1 broadcast candidate requires M == 1"
    return None


def splitk_rejection_reason(a, b, scales, config):
    reason = _common_rejection_reason(a, b, scales)
    if reason is not None:
        return reason
    if not isinstance(config, (tuple, list)) or len(config) != 6:
        return "split-k config must be (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, split_k)"
    if any(not isinstance(value, int) or value <= 0 for value in config):
        return "split-k config values must be positive integers"
    _, block_n, block_k, _, _, split_k = config
    if b.shape[0] % block_n != 0:
        return "current split-k kernel requires N divisible by BLOCK_N"
    if a.shape[1] % (block_k * split_k) != 0:
        return "current split-k kernel requires K divisible by BLOCK_K * SPLIT_K"
    return None


def separate_rejection_reason(a, b, scales):
    return _common_rejection_reason(a, b, scales)


def algorithm_workspace_bytes(a, b, algorithm, config=None):
    if algorithm == "splitk":
        split_k = config[5]
        return split_k * a.shape[0] * b.shape[0] * 4
    if algorithm == "separate":
        # Converted BF16 weight plus the BF16 unscaled GEMM output.
        return b.numel() * 2 + a.shape[0] * b.shape[0] * 2
    return 0


@libentry()
@triton.jit
def weight_int8pack_mm_dense(
    A,
    B,
    Scales,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = 8 * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * 8, 8)
    pid_m = group_id * 8 + pid % group_size
    pid_n = (pid % width) // group_size

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k = k_start + offsets_k
        a_ptrs = A + offsets_m[:, None] * stride_am + k[None, :] * stride_ak
        b_ptrs = B + offsets_n[:, None] * stride_bn + k[None, :] * stride_bk
        if EVEN_M and EVEN_K:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=(offsets_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
        if EVEN_N and EVEN_K:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(
                b_ptrs,
                mask=(offsets_n[:, None] < N) & (k[None, :] < K),
                other=0,
            )
        acc += tl.dot(a, tl.trans(b.to(tl.bfloat16)), out_dtype=tl.float32,
                      allow_tf32=False)

    n_mask = offsets_n < N
    scale = tl.load(Scales + offsets_n, mask=n_mask, other=0.0).to(tl.float32)
    output = acc * scale[None, :]
    out_ptrs = C + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(out_ptrs, output.to(tl.bfloat16),
             mask=(offsets_m[:, None] < M) & n_mask[None, :])


@libentry()
@triton.jit
def weight_int8pack_mm_m1_broadcast(
    A,
    B,
    Scales,
    C,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ak,
    stride_bn,
    stride_bk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    offsets_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offsets_k
        if EVEN_K:
            a = tl.load(A + k * stride_ak)
        else:
            a = tl.load(A + k * stride_ak, mask=k < K, other=0.0)
        a_rows = tl.broadcast_to(a[None, :], (16, BLOCK_K))
        b_ptrs = B + offsets_n[:, None] * stride_bn + k[None, :] * stride_bk
        if EVEN_N and EVEN_K:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(
                b_ptrs,
                mask=(offsets_n[:, None] < N) & (k[None, :] < K),
                other=0,
            )
        acc += tl.dot(a_rows, tl.trans(b.to(tl.bfloat16)),
                      out_dtype=tl.float32, allow_tf32=False)
    if EVEN_N:
        scale = tl.load(Scales + offsets_n).to(tl.float32)
    else:
        scale = tl.load(Scales + offsets_n, mask=offsets_n < N,
                        other=0.0).to(tl.float32)
    output = acc * scale[None, :]
    rows = tl.arange(0, 16)
    tl.store(C + rows[:, None] * N + offsets_n[None, :],
             output.to(tl.bfloat16),
             mask=(rows == 0)[:, None] & (offsets_n < N)[None, :])


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
    # This is a production policy predicate, not Separate's capability.
    return (
        a.ndim == 2
        and b.ndim == 2
        and a.shape[0] >= 113
        and b.shape[0] % _BLOCK_N == 0
        and a.shape[1] % (_BLOCK_K * _SPLIT_K) == 0
        and separate_rejection_reason(a, b, scales) is None
    )


_DOWN_M2048_SCALE_TILE = (64, 128, 128, 4, 4, "cpasync")


def _cached_mm_nt_config(provider, a, converted, output):
    """Reuse the registered BF16 NT MM choice without launching a tuner."""
    mm_entry = getattr(provider, "mm_kernel_nt", None)
    tuner = getattr(mm_entry, "fn", None)
    if tuner is None or not hasattr(tuner, "get_key") or not hasattr(tuner, "cache"):
        return None
    m, k = a.shape
    n = converted.shape[0]
    key = tuner.get_key({
        "A": a, "B": converted, "C": output,
        "M": m, "N": n, "K": k,
    })
    return tuner.cache.get(key)


def _large_m_separate(a, b, scales):
    down_m_is_eligible = (
        a.ndim == 2
        and (a.shape[0] == 2048 or _can_use_large_m_separate(a, b, scales))
    )
    if b.shape == (2048, 6144) and down_m_is_eligible:
        import importlib
        import flag_gems
        provider = importlib.import_module(flag_gems.mm.__module__)
        if hasattr(provider, "mm_kernel_nt_scale_direct"):
            fused = _large_m_separate_fused_scale(a, b, scales, direct_scale=True)
            if fused is not None:
                return fused
    # Keep scale after GEMM. MetaX MM materializes BF16 before this scale,
    # introducing one rounding point versus an FP32 fused epilogue.
    # Neither the converted weights nor the unscaled output persist.
    with torch_device_fn.device(a.device):
        converted = torch.empty(b.shape, device=b.device, dtype=torch.bfloat16)
        # Offline full-call validation on C500: larger contiguous conversion
        # blocks improve full Separate calls at the tested Down M points.
        # This is a performance policy; other shapes retain the legal default.
        convert_block = (
            4096 if b.shape == (2048, 6144) and 113 <= a.shape[0] <= 4096
            else 1024
        )
        weight_int8pack_convert[(triton.cdiv(b.numel(), convert_block),)](
            b, converted, b.numel(), BLOCK=convert_block, num_warps=4)
        # Reuse the registered provider, including opt-in tiling overrides.
        # Importing the vendor module under a second name can bypass them.
        import flag_gems
        unscaled = flag_gems.mm(a, converted.t())
        output = torch.empty_like(unscaled)
        weight_int8pack_output_scale[(triton.cdiv(output.numel(), 1024),)](
            unscaled, scales, output, output.shape[1], output.numel(), BLOCK=1024)
    return output


@lru_cache(maxsize=8)
def _down_scale_provider_matches(path):
    # Bind the measured policy to the provider used in the full-call tests.
    # Hash once during warmup; never launch a tuner during graph capture.
    if torch.__version__ != "2.8.0+metax3.7.0.7" or triton.__version__ != "3.0.0":
        return False
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() == (
            "3abfeba5c64c159370543602a2afc329efb6c7fc09e46659c11faa6b64485213"
        )
    except (OSError, TypeError):
        return False


@lru_cache(maxsize=128)
def _log_down_scale_tile(m, tile):
    # Selection evidence only: graph replay does not execute Python again.
    logger = logging.getLogger(__name__ + ".fused_scale_policy")
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.propagate = False
    logger.info("C500_W8A16_SCALE_POLICY M=%s tile=%s", m, tile)


def _down_fused_scale_tile(m):
    # Performance policy, not a kernel capability restriction. Unmeasured
    # ranges retain the registered provider's existing selection.
    if 193 <= m <= 384:
        return (64, 128, 128, 4, 4, "cpasync")
    if 385 <= m <= 16384:
        return (128, 128, 128, 4, 4, "cpasync")
    return None


def _large_m_separate_fused_scale(a, b, scales, direct_scale=False):
    """Fuse scale into the NT MM epilogue for eligible large Down calls."""
    if (_common_rejection_reason(a, b, scales) is not None
            or b.shape != (2048, 6144)
            or (a.shape[0] < 113 and a.shape[0] != 2048)):
        return None
    import importlib
    import flag_gems
    provider = importlib.import_module(flag_gems.mm.__module__)
    kernel = getattr(provider, "mm_kernel_nt_scale_direct" if direct_scale else "mm_kernel_nt_scale", None)
    if kernel is None:
        return None
    with torch_device_fn.device(a.device):
        converted = torch.empty(b.shape, device=b.device, dtype=torch.bfloat16)
        output = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=a.dtype)
        tile = None
        if (direct_scale
                and torch.cuda.get_device_name(a.device) == "MetaX C500"
                and _down_scale_provider_matches(getattr(provider, "__file__", None))):
            tile = _down_fused_scale_tile(a.shape[0])
        if tile is not None:
            _log_down_scale_tile(a.shape[0], tile)
            bm, bn, bk, warps, stages, pipeline = tile
        elif a.shape[0] == 2048:
            bm, bn, bk, warps, stages, pipeline = _DOWN_M2048_SCALE_TILE
        else:
            config = _cached_mm_nt_config(provider, a, converted, output)
            if config is None:
                return None
            try:
                bm = config.kwargs["BLOCK_M"]
                bn = config.kwargs["BLOCK_N"]
                bk = config.kwargs["BLOCK_K"]
                warps = config.num_warps
                stages = config.num_stages
                pipeline = config.kwargs["pipeline"]
            except (AttributeError, KeyError):
                return None
        weight_int8pack_convert[(triton.cdiv(b.numel(), 4096),)](
            b, converted, b.numel(), BLOCK=4096, num_warps=4)
        kernel[(triton.cdiv(a.shape[0], bm) * triton.cdiv(b.shape[0], bn),)](
            a, converted, output, scales, a.shape[0], b.shape[0], a.shape[1],
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
            EVEN_M=a.shape[0] % bm == 0, EVEN_N=b.shape[0] % bn == 0,
            EVEN_K=a.shape[1] % bk == 0,
            num_warps=warps, num_stages=stages, pipeline=pipeline)
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


@triton.jit
def weight_int8pack_mm_splitk_reduce_scale_vector(
    P, Scales, C, M, N, SPLIT_K: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    # Offline variant: load split planes together and reduce across that axis.
    # This changes FP32 addition order; callers must validate numerical error.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    splits = tl.arange(0, triton.next_power_of_2(SPLIT_K))
    values = tl.load(P + splits[:, None] * M * N + offsets[None, :],
                     mask=(splits[:, None] < SPLIT_K) &
                          (offsets[None, :] < M * N), other=0.0)
    acc = tl.sum(values, axis=0)
    scale = tl.load(Scales + offsets % N, mask=offsets < M * N, other=0.0)
    tl.store(C + offsets, (acc * scale).to(tl.bfloat16), mask=offsets < M * N)


def _can_use_small_m_splitk(a, b, scales):
    # This is the measured production policy.  The explicit Split-K entry
    # below accepts any M for which its config is legal.
    config = (16, _BLOCK_N, _BLOCK_K, 4, 2, _SPLIT_K)
    return (a.ndim == 2 and a.shape[0] in _SMALL_M_VALUES
            and splitk_rejection_reason(a, b, scales, config) is None)


def _small_m_splitk(a, b, scales):
    m, k = a.shape
    block_m = 64 if m == 64 else 16
    return _configured_splitk(
        a, b, scales, (block_m, _BLOCK_N, _BLOCK_K, 4, 2, _SPLIT_K)
    )


def _configured_splitk(a, b, scales, config, vector_reduce=False, reduce_block=256):
    reason = splitk_rejection_reason(a, b, scales, config)
    if reason is not None:
        raise ValueError(f"illegal Split-K candidate: {reason}")
    m, k = a.shape
    n = b.shape[0]
    block_m, block_n, block_k, warps, stages, split_k = config
    partials = torch.empty(
        (split_k, m, n), device=a.device, dtype=torch.float32
    )
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    partial_grid = (
        triton.cdiv(m, block_m) * triton.cdiv(n, block_n),
        split_k,
    )
    if reduce_block not in (128, 256, 512):
        raise ValueError("unsupported reduction block")
    reduce_grid = (triton.cdiv(m * n, reduce_block),)

    with torch_device_fn.device(a.device):
        # Explicit configurations vary launch options as well as constexprs.
        # LibEntry's cache key omits num_warps/num_stages, so use Triton's JIT
        # cache, which keys the compiled kernel on the complete launch options.
        weight_int8pack_mm_splitk_partial.jit_function.run(
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
            SPLIT_K=split_k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            EVEN_M=m % block_m == 0,
            num_warps=warps,
            num_stages=stages,
            pipeline="basic",
            grid=partial_grid,
            warmup=False,
        )
        reducer = (weight_int8pack_mm_splitk_reduce_scale_vector
                   if vector_reduce else weight_int8pack_mm_splitk_reduce_scale)
        reducer[reduce_grid](
            partials,
            scales,
            output,
            m,
            n,
            SPLIT_K=split_k,
            BLOCK_SIZE=reduce_block,
            num_warps=4,
        )
    return output


def _configured_dense(a, b, scales, config, m1=False, pipeline="basic"):
    # Offline exploration can select a different compiler pipeline; production
    # defaults remain unchanged until full-call and graph validation pass.
    if pipeline not in ("basic", "cpasync"):
        raise ValueError("unsupported Dense pipeline")
    reason = dense_rejection_reason(a, b, scales, config, m1=m1)
    if reason is not None:
        raise ValueError(f"illegal Dense candidate: {reason}")
    m, k = a.shape
    n = b.shape[0]
    block_m, block_n, block_k, warps, stages = config
    output = torch.empty((m, n), device=a.device, dtype=a.dtype)
    with torch_device_fn.device(a.device):
        if m1:
            weight_int8pack_mm_m1_broadcast.jit_function.run(
                a,
                b,
                scales,
                output,
                n,
                k,
                a.stride(1),
                b.stride(0),
                b.stride(1),
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                EVEN_N=n % block_n == 0,
                EVEN_K=k % block_k == 0,
                num_warps=warps,
                num_stages=stages,
                pipeline=pipeline,
                grid=(triton.cdiv(n, block_n),),
                warmup=False,
            )
        else:
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            weight_int8pack_mm_dense.jit_function.run(
                a,
                b,
                scales,
                output,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                output.stride(0),
                output.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                EVEN_M=m % block_m == 0,
                EVEN_N=n % block_n == 0,
                EVEN_K=k % block_k == 0,
                num_warps=warps,
                num_stages=stages,
                pipeline=pipeline,
                grid=grid,
                warmup=False,
            )
    return output


def official_bf16_dense_config(m, n, k):
    """Return the upstream hand-picked BF16 tile as a candidate, not policy."""
    if m <= 16:
        return (16, 64, 256, 4, 3)
    if m <= 32:
        return (32, 32, 128, 4, 3) if k >= n else (32, 64, 128, 4, 3)
    if m <= 64:
        return (64, 64, 128, 4, 3) if k >= n else (64, 128, 64, 8, 3)
    return (32, 128, 64, 2, 3)


def run_w8a16_algorithm(a, b, scales, algorithm, config=None):
    """Force one complete W8A16 algorithm for correctness and offline tuning."""
    if algorithm == "dense":
        if config is None:
            config = official_bf16_dense_config(a.shape[0], b.shape[0], a.shape[1])
        return _configured_dense(a, b, scales, config)
    if algorithm == "dense_m1":
        if config is None:
            config = M1_DENSE_CONFIGS[0]
        return _configured_dense(a, b, scales, config, m1=True)
    if algorithm == "splitk":
        if config is None:
            config = (16, _BLOCK_N, _BLOCK_K, 4, 2, _SPLIT_K)
        return _configured_splitk(a, b, scales, config)
    if algorithm == "separate":
        reason = separate_rejection_reason(a, b, scales)
        if reason is not None:
            raise ValueError(f"illegal Separate candidate: {reason}")
        return _large_m_separate(a, b, scales)
    raise ValueError(f"unknown W8A16 algorithm: {algorithm}")


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
    if os.environ.get("VLLM_FL_W8A16_SHAPE_DISPATCH") == "1":
        return _shape_dispatch(a, b, scales)
    if _can_use_small_m_splitk(a, b, scales):
        return _small_m_splitk(a, b, scales)
    if _can_use_large_m_separate(a, b, scales):
        return _large_m_separate(a, b, scales)
    return _generic_weight_int8pack_mm(a, b, scales)


@lru_cache(maxsize=1)
def _dispatch_manifest():
    path = Path(__file__).with_name("w8a16_dispatch.json")
    data = json.loads(path.read_text())
    if (data.get("schema_version") != 1
            or data.get("contract") != "BF16xINT8_to_BF16_FP32_acc_scale_after_dot"
            or data.get("device") != "MetaX C500"
            or data.get("torch_version") != torch.__version__
            or data.get("triton_version") != triton.__version__):
        raise RuntimeError("W8A16 dispatch manifest version mismatch")
    data["exact_index"] = {tuple(entry["shape"]): entry
                           for entry in data["exact"]}
    return data


def _dispatch_group(m, n, k):
    if (n, k) == (2048, 6144):
        if 1 <= m <= 16: return "down_tiny", "splitk"
        if 17 <= m <= 47: return "down_small", "splitk"
        if 48 <= m <= 64: return "down_transition", "splitk"
        if 65 <= m <= 112: return "down_upper", "splitk"
    if (n, k) == (12288, 2048):
        if 1 <= m <= 16: return "gate_up_tiny", "splitk"
        if 17 <= m <= 32: return "gate_up_small", "splitk"
        if 33 <= m <= 47: return "gate_up_dense_transition", "dense"
        if 48 <= m <= 112: return "gate_up_dense_mid", "dense"
    if (n, k) == (2560, 2048):
        if 1 <= m <= 16: return "qkv_tiny", "splitk"
        if 17 <= m <= 32: return "qkv_small", "splitk"
        if 33 <= m <= 96: return "qkv_mid", "splitk"
    if (n, k) == (2048, 2048):
        if 1 <= m <= 8: return "o_tiny", "splitk"
        if 9 <= m <= 32: return "o_small", "splitk"
        if 33 <= m <= 104: return "o_mid", "splitk"
    return None, "separate"


def _dispatch_candidate_reason(a, b, scales, algorithm, config):
    m, k = a.shape
    n = b.shape[0]
    if algorithm == "splitk":
        reason = splitk_rejection_reason(a, b, scales, config)
        if reason: return reason
        bm, bn, bk, _, _, split_k = config
        if 4 * split_k * m * n > 256 * 1024**2:
            return "workspace_budget"
    else:
        reason = dense_rejection_reason(a, b, scales, config)
        if reason: return reason
        bm, bn, bk, _, _ = config
    # This is a light performance rule; it never overrides capability checks.
    if bm > max(16, 2 * m): return "excess_M_padding"
    if bk > k: return "excess_K_padding"
    return None


def _shape_dispatch(a, b, scales):
    reason = _common_rejection_reason(a, b, scales)
    if reason is not None:
        # Preserve the upstream correctness path for unsupported layouts.
        return _generic_weight_int8pack_mm(a, b, scales)
    # Diagnostic ablation only: keep quantization coverage and the BF16 MM
    # provider fixed while removing Dense/Split-K selection. Restart the
    # model service after changing this environment variable.
    if os.environ.get("VLLM_FL_W8A16_FORCE_SEPARATE", "0") == "1":
        if separate_rejection_reason(a, b, scales) is None:
            return _large_m_separate(a, b, scales)
        return _generic_weight_int8pack_mm(a, b, scales)
    m, k = a.shape
    n = b.shape[0]
    manifest = _dispatch_manifest()
    if (m, n, k) == (64, 2048, 6144):
        return _configured_splitk(a, b, scales, (64, 64, 128, 4, 1, 8))
    group, algorithm = _dispatch_group(m, n, k)
    if group is not None:
        exact = manifest["exact_index"].get((m, n, k))
        if exact is not None and exact["algorithm"] != algorithm:
            exact = None
        candidates = ([exact["config"]] if exact else []) + manifest["pools"][group]
        for config in candidates:
            if _dispatch_candidate_reason(a, b, scales, algorithm, config) is None:
                if algorithm == "splitk":
                    return _configured_splitk(a, b, scales, tuple(config))
                return _configured_dense(a, b, scales, tuple(config))
        if group == "gate_up_dense_transition":
            for config in manifest["pools"]["gate_up_transition"]:
                if _dispatch_candidate_reason(a, b, scales, "splitk", config) is None:
                    return _configured_splitk(a, b, scales, tuple(config))
    if separate_rejection_reason(a, b, scales) is None:
        return _large_m_separate(a, b, scales)
    return _generic_weight_int8pack_mm(a, b, scales)

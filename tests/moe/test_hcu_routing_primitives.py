# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
import tilelang
from tilelang import language as T

from tile_kernels.config import is_hcu
from tile_kernels.moe._hcu_routing import philox2x32, warp32_sum, warp32_min, warp32_match_any

pytestmark = pytest.mark.skipif(not is_hcu(), reason='HCU routing primitives')


def _philox_reference(seed, sequence):
    mask = (1 << 32) - 1
    a, b, c, d = 0, 0, sequence & mask, sequence >> 32
    k0, k1 = seed & mask, seed >> 32
    for _ in range(10):
        p0, p1 = a * 0xD2511F53, c * 0xCD9E8D57
        a, b, c, d = ((p1 >> 32) ^ b ^ k0) & mask, p1 & mask, ((p0 >> 32) ^ d ^ k1) & mask, p0 & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return a, b


@tilelang.jit
def _philox_kernel(seeds, sequences, output):
    n = T.const('n')
    seeds: T.Tensor[(n,), T.int64]
    sequences: T.Tensor[(n,), T.int64]
    output: T.Tensor[(n, 2), T.int64]
    with T.Kernel(1, threads=64):
        i = T.get_thread_binding()
        if i < n:
            first = T.alloc_var(T.uint32)
            second = T.alloc_var(T.uint32)
            philox2x32(seeds[i], sequences[i], first, second)
            output[i, 0] = T.cast(first, T.int64)
            output[i, 1] = T.cast(second, T.int64)


def test_hcu_philox_known_vectors():
    # Random123 Philox4x32-10 zero-key/zero-counter known-answer vector.
    assert _philox_reference(0, 0) == (0x6627E8D5, 0xE169C58D)
    pairs = [(0, 0), (0, 1), (1, 0), (1234, 32767), (0x123456789ABCDEF, 0x102030405060708)]
    seeds = torch.tensor([s for s, _ in pairs], dtype=torch.int64, device='cuda')
    sequences = torch.tensor([q for _, q in pairs], dtype=torch.int64, device='cuda')
    output = torch.empty((len(pairs), 2), dtype=torch.int64, device='cuda')
    _philox_kernel(seeds, sequences, output)
    expected = torch.tensor([_philox_reference(s, q) for s, q in pairs], dtype=torch.int64, device='cuda')
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@tilelang.jit
def _warp32_kernel(output):
    output: T.Tensor[(64, 3), T.int64]
    with T.Kernel(1, threads=64):
        lane = T.get_thread_binding()
        total = T.alloc_var(T.int32, init=lane // 32 + 1)
        minimum = T.alloc_var(T.int32, init=T.if_then_else(lane < 32, 7, -3))
        warp32_sum(total)
        warp32_min(minimum)
        output[lane, 0] = total
        output[lane, 1] = minimum
        output[lane, 2] = T.cast(warp32_match_any(lane % 4), T.int64)


def test_hcu_warp32_isolation():
    output = torch.empty((64, 3), dtype=torch.int64, device='cuda')
    _warp32_kernel(output)
    expected = torch.tensor([
        [32 if lane < 32 else 64, 7 if lane < 32 else -3,
         sum(1 << other for other in range(32) if other % 4 == lane % 4)]
        for lane in range(64)
    ], dtype=torch.int64, device='cuda')
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

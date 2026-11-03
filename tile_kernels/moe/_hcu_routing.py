# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# SPDX-License-Identifier: MIT

"""HCU equivalents for the warp32 random-routing primitives used upstream."""

from tilelang import language as T


@T.macro
def philox2x32(seed, sequence, first: T.Ref, integer: T.Ref):
    """First two draws of Philox4x32-10 at (seed, subsequence, offset=0).

    Counter/key layout and round constants follow Random123/rocRAND Philox.
    No global state is advanced: repeated routing calls use the same stream.
    """
    c0 = T.alloc_var(T.uint32, init=0)
    c1 = T.alloc_var(T.uint32, init=0)
    c2 = T.alloc_var(T.uint32, init=T.cast(sequence, T.uint32))
    c3 = T.alloc_var(T.uint32, init=T.cast(T.cast(sequence, T.uint64) >> 32, T.uint32))
    k0 = T.alloc_var(T.uint32, init=T.cast(seed, T.uint32))
    k1 = T.alloc_var(T.uint32, init=T.cast(T.cast(seed, T.uint64) >> 32, T.uint32))
    for _ in T.unroll(10):
        p0 = T.alloc_var(T.uint64)
        p1 = T.alloc_var(T.uint64)
        next0 = T.alloc_var(T.uint32)
        next2 = T.alloc_var(T.uint32)
        p0 = T.cast(c0, T.uint64) * T.cast(0xD2511F53, T.uint64)
        p1 = T.cast(c2, T.uint64) * T.cast(0xCD9E8D57, T.uint64)
        next0 = T.cast(p1 >> 32, T.uint32) ^ c1 ^ k0
        next2 = T.cast(p0 >> 32, T.uint32) ^ c3 ^ k1
        c0 = next0
        c1 = T.cast(p1, T.uint32)
        c2 = next2
        c3 = T.cast(p0, T.uint32)
        k0 += T.cast(0x9E3779B9, T.uint32)
        k1 += T.cast(0xBB67AE85, T.uint32)
    first = c0
    integer = c1


@T.macro
def philox_uniform_and_uint(seed, sequence, uniform: T.Ref, integer: T.Ref):
    first = T.alloc_var(T.uint32)
    philox2x32(seed, sequence, first, integer)
    uniform = T.cast(first, T.float32) * (2.0**-32) + (2.0**-32)


@T.macro
def warp32_sum(value: T.Ref):
    for i in T.unroll(5):
        value += T.shfl_xor(value, 1 << (4 - i), width=32)


@T.macro
def warp32_min(value: T.Ref):
    for i in T.unroll(5):
        value = T.min(value, T.shfl_xor(value, 1 << (4 - i), width=32))


@T.macro
def warp32_match_any(value):
    # HCU's native match helper returns a 64-lane mask. Restrict it to this
    # logical half-wave, then shift back to the upstream 32-bit mask layout.
    base = (T.get_thread_binding() % 64) // 32 * 32
    mask = T.cast(0xFFFFFFFF, T.uint64) << base
    matches = T.call_extern(T.uint64, 'tl::match_any_sync', mask, value)
    return T.cast(matches >> base, T.uint32)

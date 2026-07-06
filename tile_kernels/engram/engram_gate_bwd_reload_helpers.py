# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# SPDX-License-Identifier: MIT

"""Helpers for estimating engram_gate_bwd SMEM and choosing tiling for the reload (low-LDS) backward kernel."""

import functools
import math

import torch

from tile_kernels.config import get_max_smem_per_sm


def _engram_bwd_warp_layout():
    warp_size = 64 if torch.version.hip is not None else 32
    warps_per_head = 1 if torch.version.hip is not None else 2
    return warp_size, warps_per_head


def engram_gate_bwd_v_vec_size(elems_per_thread: int) -> int:
    """Pick largest ``vec ∈ {4,2,1}`` with ``elems_per_thread % vec == 0`` (grad_v tiling)."""
    for vec in (4, 2, 1):
        if elems_per_thread % vec == 0:
            return vec

    return 1


def engram_gate_bwd_go_vec_size(hidden_size: int, threads_per_head: int) -> int:
    """Pick largest ``vec ∈ {8,4,2,1}`` with ``hidden_size % (threads_per_head * vec) == 0``."""

    for vec in (8, 4, 2, 1):
        go_tile = threads_per_head * vec
        if hidden_size % go_tile == 0:
            return vec

    raise ValueError(f'No valid go_vec_size for hidden_size={hidden_size}, threads_per_head={threads_per_head}')


@functools.lru_cache(maxsize=None)
def estimate_engram_gate_bwd_reload_pipeline_smem_bytes(hidden_size: int, hc_mult: int = 4) -> int:
    """Static SMEM footprint (bytes) of ``get_engram_gate_bwd_kernel`` allocations."""

    def _choose_go_blk_d(hs, go_tile):
        result = go_tile
        for blk in range(go_tile, hs // 2 + 1, go_tile):
            if hs % blk == 0:
                result = blk
        return result

    def _choose_x_blk_d(hs, x_tile_inner, hc, warps_ph):
        from tile_kernels.config import get_max_smem_per_sm as _gsm

        smem_fixed_inner = (hc + 1) * hs * 2 + hc * warps_ph * 4
        smem_per_x = 2 * hc * (2 + 2 + 4)
        x_smem_limit = (_gsm() - smem_fixed_inner) // smem_per_x
        x_limit = min(x_smem_limit, hs // 2)
        result = x_tile_inner
        for blk in range(x_tile_inner, x_limit + 1, x_tile_inner):
            if hs % blk == 0:
                result = blk
        return result

    warp_size, warps_per_head = _engram_bwd_warp_layout()
    threads_per_head = warp_size * warps_per_head
    threads = hc_mult * threads_per_head
    if hidden_size % threads != 0:
        return 1 << 62
    elems_per_thread = hidden_size // threads
    x_vec_size = 4
    v_vec_size = engram_gate_bwd_v_vec_size(elems_per_thread)
    go_vec_size = engram_gate_bwd_go_vec_size(hidden_size, threads_per_head)
    go_blk_d = _choose_go_blk_d(hidden_size, threads_per_head * go_vec_size)
    x_blk_d = _choose_x_blk_d(hidden_size, threads_per_head * x_vec_size, hc_mult, warps_per_head)

    go_smem_b = hc_mult * hidden_size * 2
    v_smem_b = hidden_size * 2
    xkw_b = 2 * hc_mult * x_blk_d * (2 + 2 + 4)
    dldg_b = hc_mult * warps_per_head * 4
    return go_smem_b + v_smem_b + xkw_b + dldg_b


@functools.lru_cache(maxsize=None)
def engram_gate_bwd_reload_pipeline_exceeds_smem_budget(
    hidden_size: int, hc_mult: int = 4, budget_bytes: int | None = None
) -> bool:
    if budget_bytes is None:
        budget_bytes = get_max_smem_per_sm()
    return estimate_engram_gate_bwd_reload_pipeline_smem_bytes(hidden_size, hc_mult) > budget_bytes


def _reload_h_blk_alignment(hidden_size: int, hc_mult: int) -> int:
    """LCM of subtiles needed by reload-path dldg, pass2, and grad_v within a slice."""
    warp_size, warps_per_head = _engram_bwd_warp_layout()
    threads_per_head = warp_size * warps_per_head
    threads = hc_mult * threads_per_head
    if hidden_size % threads != 0:
        raise ValueError(f'hidden_size % threads != 0: {hidden_size} % {threads}')
    elems_per_thread = hidden_size // threads
    x_vec_size = 4
    v_vec_size = engram_gate_bwd_v_vec_size(elems_per_thread)
    go_vec_size = engram_gate_bwd_go_vec_size(hidden_size, threads_per_head)
    a = threads_per_head * go_vec_size
    b = threads_per_head * x_vec_size
    c = threads * v_vec_size
    lcm_ab = a // math.gcd(a, b) * b
    return lcm_ab // math.gcd(lcm_ab, c) * c


RELOAD_PASS2_X_VEC_SIZE = 4


def _reload_pass2_h_blk_alignment(hc_mult: int) -> int:
    """Pass2 uses ``x_vec_size=4`` copy/compute subtiles."""
    warp_size, warps_per_head = _engram_bwd_warp_layout()
    threads_per_head = warp_size * warps_per_head
    return threads_per_head * RELOAD_PASS2_X_VEC_SIZE


def _reload_pass1_pingpong_bytes(h_blk: int, hc_mult: int) -> int:
    return 2 * (hc_mult * h_blk * 2 + h_blk * 2)


def _reload_pass2_pingpong_bytes(h_blk: int, hc_mult: int) -> int:
    single = hc_mult * h_blk * (2 + 2 + 2 + 4)
    return 2 * single


@functools.lru_cache(maxsize=None)
def estimate_engram_gate_bwd_reload_pingpong_smem_bytes(
    hidden_size: int, hc_mult: int = 4, budget_bytes: int | None = None
) -> int:
    """Conservative SMEM estimate for reload ping-pong (Pass1 + Pass2 buffers, no merge)."""
    if budget_bytes is None:
        budget_bytes = get_max_smem_per_sm()
    _, h_blk1 = choose_reload_hidden_block(hidden_size, hc_mult)
    h_blk2 = choose_reload_pass2_hidden_block(hidden_size, hc_mult, budget_bytes=budget_bytes)
    dldg_b = hc_mult * _engram_bwd_warp_layout()[1] * 4
    pass1 = _reload_pass1_pingpong_bytes(h_blk1, hc_mult)
    pass2 = _reload_pass2_pingpong_bytes(h_blk2, hc_mult)
    return pass1 + pass2 + dldg_b


def reload_pass1_prefetch_wait_groups(
    h_blk1: int,
    threads_per_head: int,
    go_vec_size: int,
    threads: int,
    v_vec_size: int,
) -> int:
    """``ptx_wait_group(N)`` for one Pass1 ``go_bb + v_bb`` prefetch batch (HCU).

    ``go`` (Fragment, ``go_vec``) lowers to ``go_sub`` × ``cp_async_gs<go_vec*2>``.
    ``v`` (flat) lowers to one ``cp_async_gs<v_vec*2>`` per slice (``v_groups``).
    """

    go_sub = h_blk1 // (threads_per_head * go_vec_size)
    v_groups = h_blk1 // (threads * v_vec_size)
    return go_sub + v_groups


def reload_pass2_prefetch_wait_groups(x_sub_blks: int) -> int:
    """``ptx_wait_group(N)`` for one Pass2 ``go2/xh/kh/wf`` prefetch batch (HCU).

    Four buffers; each ``T.async_copy`` lowers to ``x_sub_blks`` async ops
    (``go2/xh/kh`` → ``cp_async_gs<8>``, ``wf_bb`` fp32 → ``cp_async_gs<16>``).
    """

    return 4 * x_sub_blks


@functools.lru_cache(maxsize=None)
def choose_reload_hidden_block(hidden_size: int, hc_mult: int = 4):
    """Return ``(threads, h_blk_pass1)`` for reload Pass1 (dldg + grad_v).

    ``h_blk_pass1`` must divide ``hidden_size``, be a multiple of the pipeline-aligned subtile
    (including ``threads * v_vec`` for bounded ``grad_v`` over slices).
    Pass1 uses ping-pong ``go``/``v`` buffers sized to this tile.
    """

    warp_size, warps_per_head = _engram_bwd_warp_layout()
    threads_per_head = warp_size * warps_per_head
    threads = hc_mult * threads_per_head
    align_tile = _reload_h_blk_alignment(hidden_size, hc_mult)

    candidates = []
    for h_try in (256, 384, 512, 768, 1024):
        blk = math.gcd(hidden_size, h_try)
        if blk > 0 and hidden_size % blk == 0 and blk % align_tile == 0:
            tiles = hidden_size // blk
            if tiles >= 2:
                candidates.append((blk, threads))
    if not candidates:
        blk = align_tile
        while blk <= hidden_size // 2:
            if hidden_size % blk == 0 and blk % align_tile == 0:
                tiles = hidden_size // blk
                if tiles >= 2:
                    return threads, blk
            blk += align_tile
        raise ValueError(
            f'Cannot pick reload Pass1 tile for hidden_size={hidden_size} (need multiple of {align_tile})'
        )
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1], candidates[0][0]


@functools.lru_cache(maxsize=None)
def choose_reload_pass2_hidden_block(
    hidden_size: int, hc_mult: int = 4, budget_bytes: int | None = None
) -> int:
    """Return ``h_blk_pass2`` for reload Pass2 ping-pong (``x_vec_size=4``).

    Prefer ``512`` when it divides ``hidden_size`` and the conservative no-merge footprint
    fits ``budget_bytes``; otherwise pick the largest valid tile that fits.
    """
    if budget_bytes is None:
        budget_bytes = get_max_smem_per_sm()
    _, h_blk1 = choose_reload_hidden_block(hidden_size, hc_mult)
    pass1_pp = _reload_pass1_pingpong_bytes(h_blk1, hc_mult)
    dldg_b = hc_mult * _engram_bwd_warp_layout()[1] * 4
    align_tile = _reload_pass2_h_blk_alignment(hc_mult)

    candidates = []
    for h_try in (512, 768, 1024, 256):
        blk = math.gcd(hidden_size, h_try)
        if blk > 0 and hidden_size % blk == 0 and blk % align_tile == 0:
            tiles = hidden_size // blk
            if tiles >= 2:
                pass2_pp = _reload_pass2_pingpong_bytes(blk, hc_mult)
                worst = pass1_pp + pass2_pp + dldg_b
                if worst <= budget_bytes:
                    candidates.append(blk)
    if not candidates:
        blk = align_tile
        while blk <= hidden_size // 2:
            if hidden_size % blk == 0 and blk % align_tile == 0:
                tiles = hidden_size // blk
                if tiles >= 2:
                    pass2_pp = _reload_pass2_pingpong_bytes(blk, hc_mult)
                    if pass1_pp + pass2_pp + dldg_b <= budget_bytes:
                        return blk
            blk += align_tile
        raise ValueError(
            f'Cannot pick reload Pass2 tile for hidden_size={hidden_size} within {budget_bytes} B'
        )
    if 512 in candidates:
        return 512
    return max(candidates)

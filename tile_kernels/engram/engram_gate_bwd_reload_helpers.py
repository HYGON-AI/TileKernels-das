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


@functools.lru_cache(maxsize=None)
def choose_reload_hidden_block(hidden_size: int, hc_mult: int = 4):
    """Return ``(threads, h_blk)`` for the reload path: gcd-based tiles with strict alignment.

    ``h_blk`` must divide ``hidden_size``, be a multiple of the pipeline-aligned subtile
    (including ``threads * v_vec`` for bounded ``grad_v`` over slices).
    Scratch peak: ``go+v`` (Pass1-in-loop) versus ``go+x+k+w+dldg`` (Pass2-in-loop).

    Uses bf16 tiles only (same as pipeline ``grad_v``: no FP32 gv column buffer).

    """

    warp_size, warps_per_head = _engram_bwd_warp_layout()
    threads_per_head = warp_size * warps_per_head
    threads = hc_mult * threads_per_head
    align_tile = _reload_h_blk_alignment(hidden_size, hc_mult)
    dldg_smem_reload_b = hc_mult * warps_per_head * 4

    candidates = []
    for h_try in (256, 384, 512, 768, 1024):
        blk = math.gcd(hidden_size, h_try)
        if blk > 0 and hidden_size % blk == 0 and blk % align_tile == 0:
            tiles = hidden_size // blk
            if tiles >= 2:
                go_v_peak = hc_mult * blk * 2 + blk * 2
                peak2 = blk * hc_mult * 2 + 2 * blk * hc_mult * 2 + blk * hc_mult * 4
                smem_peak = max(go_v_peak, peak2) + dldg_smem_reload_b
                candidates.append((smem_peak, threads, blk))
    if not candidates:
        blk = align_tile
        while blk <= hidden_size // 2:
            if hidden_size % blk == 0 and blk % align_tile == 0:
                tiles = hidden_size // blk
                if tiles >= 2:
                    return threads, blk
            blk += align_tile
        raise ValueError(
            f'Cannot pick reload tile for hidden_size={hidden_size} (need multiple of {align_tile})'
        )
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    return best[1], best[2]

import os
from functools import partial
from typing import Optional, Tuple

import torch
import tilelang
from tilelang import language as T

from tile_kernels.config import get_max_smem_per_sm, get_num_sms
from tile_kernels.engram.engram_gate_bwd_reload_helpers import (
    RELOAD_PASS2_X_VEC_SIZE,
    choose_reload_hidden_block,
    choose_reload_pass2_hidden_block,
    engram_gate_bwd_go_vec_size,
    engram_gate_bwd_reload_pipeline_exceeds_smem_budget,
    engram_gate_bwd_v_vec_size,
    reload_pass1_prefetch_wait_groups,
    reload_pass2_prefetch_wait_groups,
)


def _get_positive_env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value in (None, ''):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f'{name} must be positive, got {parsed}')
    return parsed


def _get_nonnegative_env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value in (None, ''):
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f'{name} must be non-negative, got {parsed}')
    return parsed


def _choose_engram_fwd_threads_vec_size(blk_d: int) -> tuple[int, int]:
    """CUDA warp32 vs HIP wave64; choose vec_size so ``blk_d % (threads * vec_size) == 0``.

    Tries ``vec_size ∈ {8, 4, 2, 1}`` in order (same default preference as before).
    """
    threads = 64 if torch.version.hip is not None else 32
    for vec_size in (8, 4, 2, 1):
        reduce_blk = threads * vec_size
        if blk_d % reduce_blk == 0:
            return threads, vec_size
    raise ValueError(f'engram fwd: blk_d={blk_d} incompatible with threads={threads} for vec_size in (8,4,2,1)')


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def get_engram_gate_fwd_kernel(
    hidden_size: int,
    eps: float,
    scalar: float,
    k_stride_s: int,
    k_stride_h: int,
    v_stride_s: int,
    num_sms: int,
    clamp_value: float = 1e-6,
    hc_mult: int = 4,
    save_for_backward: bool = True,
):
    """Forward kernel. When save_for_backward=True, saves dot/gate_score/rstd_x/rstd_k for backward."""
    num_tokens = T.dynamic('num_tokens')

    # NOTE Performance only tuned for hidden_size in {4096, 7168}
    def _choose_blk_d(hidden_size):
        for blk in [1024, 768, 512, 256]:
            if hidden_size % blk == 0 and hidden_size >= 2 * blk:
                return blk
        raise ValueError(f'No valid blk_d for hidden_size={hidden_size}')

    def _choose_num_persistent_blocks(hidden_size, blk_d, num_sms, hc_mult):
        """Estimate from SM shared memory occupancy, capped by register pressure."""
        # smem per block: x_smem (bf16) + kv_smem double-buffer (bf16)
        smem_bytes = hidden_size * 2 + blk_d * 4
        blocks_per_sm = min(get_max_smem_per_sm() // smem_bytes, 16)  # 16: register pressure cap
        return num_sms * blocks_per_sm // hc_mult

    blk_d = _choose_blk_d(hidden_size)
    threads, vec_size = _choose_engram_fwd_threads_vec_size(blk_d)
    num_persistent_blocks = _choose_num_persistent_blocks(hidden_size, blk_d, num_sms, hc_mult)

    assert hidden_size % blk_d == 0
    assert hidden_size >= 2 * blk_d
    num_blk = hidden_size // blk_d
    reduce_blk = threads * vec_size
    sub_blks = blk_d // reduce_blk
    v_start_phase = num_blk % 2

    @T.prim_func
    def engram_gate_fwd_kernel(
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        k: T.StridedTensor[(num_tokens, hc_mult, hidden_size), (k_stride_s, k_stride_h, 1), T.bfloat16],
        v: T.StridedTensor[(num_tokens, hidden_size), (v_stride_s, 1), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float],
        output: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        dot_out: T.Tensor[(num_tokens, hc_mult), T.float],
        gate_score: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_x: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_k: T.Tensor[(num_tokens, hc_mult), T.float],
    ):
        with T.Kernel(hc_mult, num_persistent_blocks, threads=threads) as (pid_h, pid_b):
            thread_idx = T.get_thread_binding()
            x_local = T.alloc_local((vec_size,), T.float)
            k_local = T.alloc_local((vec_size,), T.float)
            w_local = T.alloc_local((vec_size,), T.float)
            v_local = T.alloc_local((vec_size,), T.float)
            gate_score_local = T.alloc_local((1,), T.float)
            rstd_x_local = T.alloc_local((1,), T.float)
            rstd_k_local = T.alloc_local((1,), T.float)
            gate_score_reducer = T.alloc_local((1,), T.float)
            rstd_x_reducer = T.alloc_local((1,), T.float)
            rstd_k_reducer = T.alloc_local((1,), T.float)

            x_smem = T.alloc_shared((hidden_size,), T.bfloat16)
            kv_smem = T.alloc_shared((2, blk_d), T.bfloat16)

            per_block = T.ceildiv(num_tokens, num_persistent_blocks)
            t_start = T.min(per_block * pid_b, num_tokens)
            t_end = T.min(per_block * (pid_b + 1), num_tokens)

            for i_s in T.Serial(t_start, t_end):
                # === Pass 1: Reduction with cp.async pipeline ===
                if i_s == t_start:
                    T.async_copy(hidden_states[i_s, pid_h, 0:blk_d], x_smem[0:blk_d])
                    T.async_copy(k[i_s, pid_h, 0:blk_d], kv_smem[0, :])

                T.clear(rstd_k_local)
                T.clear(rstd_x_local)
                T.clear(gate_score_local)

                for i_b in T.Serial(1, num_blk):
                    phase = i_b % 2
                    prev_phase = (i_b - 1) % 2
                    T.async_copy(hidden_states[i_s, pid_h, i_b * blk_d:(i_b + 1) * blk_d], x_smem[i_b * blk_d:(i_b + 1) * blk_d])
                    T.async_copy(k[i_s, pid_h, i_b * blk_d:(i_b + 1) * blk_d], kv_smem[phase, :])
                    T.ptx_wait_group(2 * sub_blks)
                    for i_sub in T.Serial(sub_blks):
                        sub_base = (i_b - 1) * blk_d + i_sub * reduce_blk
                        for i_k in T.vectorized(vec_size):
                            x_local[i_k] = x_smem[sub_base + thread_idx * vec_size + i_k]
                            k_local[i_k] = kv_smem[prev_phase, i_sub * reduce_blk + thread_idx * vec_size + i_k]
                        for i_k in T.vectorized(vec_size):
                            w_local[i_k] = weight_fused[pid_h, sub_base + thread_idx * vec_size + i_k]
                        for i_k in T.serial(vec_size):
                            rstd_x_local[0] += x_local[i_k] * x_local[i_k]
                            rstd_k_local[0] += k_local[i_k] * k_local[i_k]
                            gate_score_local[0] += x_local[i_k] * w_local[i_k] * k_local[i_k]

                # Epilogue: process last tile
                T.ptx_wait_group(0)

                # Prefetch v[0] into freed kv_smem bank
                T.async_copy(v[i_s, 0:blk_d], kv_smem[v_start_phase, :])

                for i_sub in T.Serial(sub_blks):
                    sub_base = (num_blk - 1) * blk_d + i_sub * reduce_blk
                    for i_k in T.vectorized(vec_size):
                        x_local[i_k] = x_smem[sub_base + thread_idx * vec_size + i_k]
                        k_local[i_k] = kv_smem[(num_blk - 1) % 2, i_sub * reduce_blk + thread_idx * vec_size + i_k]
                    for i_k in T.vectorized(vec_size):
                        w_local[i_k] = weight_fused[pid_h, sub_base + thread_idx * vec_size + i_k]
                    for i_k in T.serial(vec_size):
                        rstd_x_local[0] += x_local[i_k] * x_local[i_k]
                        rstd_k_local[0] += k_local[i_k] * k_local[i_k]
                        gate_score_local[0] += x_local[i_k] * w_local[i_k] * k_local[i_k]

                # Prefetch v[1]
                T.async_copy(v[i_s, blk_d:2 * blk_d], kv_smem[1 - v_start_phase, :])

                rstd_k_reducer[0] = T.warp_reduce_sum(rstd_k_local[0])
                rstd_x_reducer[0] = T.warp_reduce_sum(rstd_x_local[0])
                gate_score_reducer[0] = T.warp_reduce_sum(gate_score_local[0])

                rstd_x_reducer[0] = T.rsqrt(rstd_x_reducer[0] / hidden_size + eps)
                rstd_k_reducer[0] = T.rsqrt(rstd_k_reducer[0] / hidden_size + eps)

                if save_for_backward:
                    if thread_idx == 0:
                        dot_out[i_s, pid_h] = gate_score_reducer[0]
                        rstd_x[i_s, pid_h] = rstd_x_reducer[0]
                        rstd_k[i_s, pid_h] = rstd_k_reducer[0]

                gate_score_reducer[0] = gate_score_reducer[0] * rstd_x_reducer[0] * rstd_k_reducer[0] * scalar

                gate_score_reducer[0] = T.sigmoid(T.copysign(T.sqrt(T.clamp(T.abs(gate_score_reducer[0]), clamp_value, float('inf'))), gate_score_reducer[0]))

                if save_for_backward:
                    if thread_idx == 0:
                        gate_score[i_s, pid_h] = gate_score_reducer[0]

                # === Pass 2: Output — x from smem, v from kv_smem (tiles 0,1 already prefetched) ===
                for i_b in T.Serial(num_blk):
                    tile_phase = (v_start_phase + i_b) % 2
                    if i_b < num_blk - 1:
                        T.ptx_wait_group(sub_blks)
                    else:
                        T.ptx_wait_group(0)
                        # Prefetch next token's k and x
                        if i_s + 1 < t_end:
                            T.async_copy(k[i_s + 1, pid_h, 0:blk_d], kv_smem[0, :])
                            T.async_copy(hidden_states[i_s + 1, pid_h, 0:blk_d], x_smem[0:blk_d])
                    for i_sub in T.Serial(sub_blks):
                        sub_base = i_b * blk_d + i_sub * reduce_blk
                        for i_k in T.vectorized(vec_size):
                            x_local[i_k] = x_smem[sub_base + thread_idx * vec_size + i_k]
                            v_local[i_k] = kv_smem[tile_phase, i_sub * reduce_blk + thread_idx * vec_size + i_k]
                        for i_k in T.vectorized(vec_size):
                            output[i_s, pid_h, sub_base + thread_idx * vec_size + i_k] = x_local[i_k] + gate_score_reducer[0] * v_local[i_k]
                    # Prefetch v[i_b+2] into freed kv_smem bank
                    if i_b + 2 < num_blk:
                        T.async_copy(v[i_s, (i_b + 2) * blk_d:(i_b + 3) * blk_d], kv_smem[tile_phase, :])

    return engram_gate_fwd_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
    },
)
def get_engram_gate_bwd_kernel(
    hidden_size: int,
    scalar: float,
    k_stride_s: int,
    k_stride_h: int,
    v_stride_s: int,
    num_persistent_blocks: int,
    clamp_value: float = 1e-6,
    hc_mult: int = 4,
):
    """Backward kernel.

    grad_out and v cached in shared memory. grad_w accumulates in registers.
    Cross-warp dldg reduction via shared memory.
    """
    assert hc_mult == 4
    num_tokens = T.dynamic('num_tokens')
    # Keep logical warp size aligned with backend implementation:
    # CUDA warp=32, HCU wavefront=64.
    warp_size = 64 if torch.version.hip is not None else 32
    # CUDA uses 2 warps/head (2 * 32 = 64 threads/head).
    # HIP uses 1 wavefront/head (1 * 64 = 64 threads/head), which keeps
    # per-head parallelism while avoiding excessive shared-memory usage.
    warps_per_head = 1 if torch.version.hip is not None else 2
    num_warps = hc_mult * warps_per_head
    threads = warp_size * num_warps
    threads_per_head = warp_size * warps_per_head
    assert hidden_size % threads == 0
    elems_per_thread = hidden_size // threads
    elems_per_warp_pair = hidden_size // threads_per_head
    x_vec_size = 4

    # NOTE Performance only tuned for hidden_size in {4096, 7168}
    def _choose_v_vec_size(elems_per_thread):
        for vec in [4, 2, 1]:
            if elems_per_thread % vec == 0:
                return vec

    def _choose_go_blk_d(hidden_size, go_tile):
        """Largest multiple of go_tile that divides hidden_size, with >= 2 tiles."""
        result = go_tile
        for blk in range(go_tile, hidden_size // 2 + 1, go_tile):
            if hidden_size % blk == 0:
                result = blk
        return result

    def _choose_go_vec_size(hidden_size, threads_per_head):
        for vec in [8, 4, 2, 1]:
            go_tile = threads_per_head * vec
            if hidden_size % go_tile == 0:
                return vec
        raise ValueError(f'No valid go_vec_size for hidden_size={hidden_size}, threads_per_head={threads_per_head}')

    def _choose_x_blk_d(hidden_size, x_tile, hc_mult, warps_per_head):
        """Largest multiple of x_tile that divides hidden_size, <= hidden_size // 2, fits in smem."""
        smem_fixed = (hc_mult + 1) * hidden_size * 2 + hc_mult * warps_per_head * 4
        smem_per_x = 2 * hc_mult * (2 + 2 + 4)
        x_smem_limit = (get_max_smem_per_sm() - smem_fixed) // smem_per_x
        x_limit = min(x_smem_limit, hidden_size // 2)
        result = x_tile
        for blk in range(x_tile, x_limit + 1, x_tile):
            if hidden_size % blk == 0:
                result = blk
        return result

    v_vec_size = _choose_v_vec_size(elems_per_thread)
    go_vec_size = _choose_go_vec_size(hidden_size, threads_per_head)
    go_blk_d = _choose_go_blk_d(hidden_size, threads_per_head * go_vec_size)
    x_blk_d = _choose_x_blk_d(hidden_size, threads_per_head * x_vec_size, hc_mult, warps_per_head)

    assert hidden_size % go_blk_d == 0 and hidden_size % x_blk_d == 0
    assert go_blk_d % (threads_per_head * go_vec_size) == 0
    assert x_blk_d % (threads_per_head * x_vec_size) == 0
    assert go_blk_d + x_blk_d <= hidden_size

    def smem_layout(i, j, vs):
        thread_id = i * threads_per_head + (j // vs) % threads_per_head
        local_id = (j % vs) + (j // (threads_per_head * vs)) * vs
        return thread_id, local_id

    @T.prim_func
    def engram_gate_bwd_kernel(
        grad_out: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        k: T.StridedTensor[(num_tokens, hc_mult, hidden_size), (k_stride_s, k_stride_h, 1), T.bfloat16],
        v: T.StridedTensor[(num_tokens, hidden_size), (v_stride_s, 1), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float],
        dot_in: T.Tensor[(num_tokens, hc_mult), T.float],
        gate_in: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_x_in: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_k_in: T.Tensor[(num_tokens, hc_mult), T.float],
        grad_x: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        grad_k: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        grad_v: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
        grad_w_partial: T.Tensor[(num_persistent_blocks, hc_mult, hidden_size), T.float],
    ):
        with T.Kernel(num_persistent_blocks, threads=threads) as pid_p:
            tid = T.get_thread_binding()
            warp_id = T.get_warp_idx()
            lane_id = T.get_lane_idx()
            head_id = warp_id // warps_per_head
            sub_warp_id = warp_id % warps_per_head

            go_local = T.alloc_local((go_vec_size,), T.float)
            v_local = T.alloc_local((go_vec_size,), T.float)
            go_v_local = T.alloc_local((v_vec_size,), T.float)
            go_x_local = T.alloc_local((x_vec_size,), T.float)
            x_local = T.alloc_local((x_vec_size,), T.float)
            k_local = T.alloc_local((x_vec_size,), T.float)
            w_fused_local = T.alloc_local((x_vec_size,), T.float)
            grad_v_partial = T.alloc_local((v_vec_size,), T.float)
            grad_w_local = T.alloc_local((elems_per_warp_pair,), T.float)

            dldg_local = T.alloc_local((1,), T.float)
            dldg_r = T.alloc_local((1,), T.float)

            gate_local = T.alloc_local((hc_mult,), T.float)
            gate_local_hc = T.alloc_var(T.float)
            rstd_x_local = T.alloc_var(T.float)
            rstd_k_local = T.alloc_var(T.float)
            dot_in_local = T.alloc_var(T.float)
            dot_x_local = T.alloc_var(T.float)
            dot_k_local = T.alloc_var(T.float)

            go_copy_layout = T.Fragment((hc_mult, go_blk_d), forward_fn=partial(smem_layout, vs=go_vec_size))
            x_copy_layout = T.Fragment((hc_mult, x_blk_d), forward_fn=partial(smem_layout, vs=x_vec_size))

            go_smem = T.alloc_shared((hc_mult, hidden_size), T.bfloat16)
            v_smem = T.alloc_shared((hidden_size,), T.bfloat16)
            x_smem = T.alloc_shared((2, hc_mult, x_blk_d), T.bfloat16)
            k_smem = T.alloc_shared((2, hc_mult, x_blk_d), T.bfloat16)
            w_smem = T.alloc_shared((2, hc_mult, x_blk_d), T.float)
            dldg_smem = T.alloc_shared((hc_mult, warps_per_head), T.float)

            per_block = T.ceildiv(num_tokens, num_persistent_blocks)
            t_start = T.min(per_block * pid_p, num_tokens)
            t_end = T.min(per_block * (pid_p + 1), num_tokens)

            T.clear(grad_w_local)

            for i_s in T.serial(t_start, t_end):
                # === Prologue: load grad_out and v into smem ===
                go_sub_blks = go_blk_d // (threads_per_head * go_vec_size)
                if i_s == t_start:
                    T.async_copy(v[i_s, :], v_smem)
                    T.async_copy(grad_out[i_s, :, :go_blk_d], go_smem[:, :go_blk_d], loop_layout=go_copy_layout)
                    T.ptx_wait_group(go_sub_blks)
                    # ensure v_smem is readable by all threads
                    T.sync_threads()

                T.copy(gate_in[i_s, :], gate_local)

                # Per-head scalar loads
                gate_local_hc = gate_in[i_s, head_id]
                rstd_x_local = rstd_x_in[i_s, head_id]
                rstd_k_local = rstd_k_in[i_s, head_id]
                dot_in_local = dot_in[i_s, head_id]

                T.clear(dldg_local)
                num_go_tiles = hidden_size // go_blk_d

                # === Pass 1a: dldg — two warps per head ===
                for i_b in T.serial(1, num_go_tiles):
                    T.async_copy(grad_out[i_s, :, i_b * go_blk_d:(i_b + 1) * go_blk_d],
                                 go_smem[:, i_b * go_blk_d:(i_b + 1) * go_blk_d],
                                 loop_layout=go_copy_layout)
                    T.ptx_wait_group(go_sub_blks)
                    for i_sub in T.serial(go_sub_blks):
                        go_base = (i_b - 1) * go_blk_d + i_sub * threads_per_head * go_vec_size + sub_warp_id * warp_size * go_vec_size
                        for i_k in T.vectorized(go_vec_size):
                            go_local[i_k] = go_smem[head_id, go_base + lane_id * go_vec_size + i_k]
                            v_local[i_k] = v_smem[go_base + lane_id * go_vec_size + i_k]
                        for i_k in T.serial(go_vec_size):
                            dldg_local[0] += go_local[i_k] * v_local[i_k]

                # Epilogue: process last go tile
                T.ptx_wait_group(0)
                for i_sub in T.serial(go_sub_blks):
                    go_base = (num_go_tiles - 1) * go_blk_d + i_sub * threads_per_head * go_vec_size + sub_warp_id * warp_size * go_vec_size
                    for i_k in T.vectorized(go_vec_size):
                        go_local[i_k] = go_smem[head_id, go_base + lane_id * go_vec_size + i_k]
                        v_local[i_k] = v_smem[go_base + lane_id * go_vec_size + i_k]
                    for i_k in T.serial(go_vec_size):
                        dldg_local[0] += go_local[i_k] * v_local[i_k]

                # Cross-warp dldg reduction via smem
                dldg_local[0] = T.warp_reduce_sum(dldg_local[0])
                if lane_id == 0:
                    dldg_smem[head_id, sub_warp_id] = dldg_local[0]
                T.sync_threads()

                # Prefetch next token's v
                if i_s + 1 < t_end:
                    T.async_copy(v[i_s + 1, :], v_smem)

                T.async_copy(hidden_states[i_s, :, :x_blk_d], x_smem[0, :, :], loop_layout=x_copy_layout)
                T.async_copy(k[i_s, :, :x_blk_d], k_smem[0, :, :], loop_layout=x_copy_layout)
                T.async_copy(weight_fused[:, :x_blk_d], w_smem[0, :, :], loop_layout=x_copy_layout)

                # Gate derivative
                T.clear(dldg_r)
                for i_w in T.unroll(warps_per_head):
                    dldg_r[0] += dldg_smem[head_id, i_w]
                dldg_r[0] = T.Select(
                    T.abs(dot_in_local) * scalar * rstd_x_local * rstd_k_local < clamp_value,
                    0.0,
                    dldg_r[0] * gate_local_hc * (1.0 - gate_local_hc) * 0.5 * T.sqrt(scalar * rstd_x_local * rstd_k_local / T.abs(dot_in_local))
                )

                # === Pass 1b: grad_v ===
                for i in T.serial(elems_per_thread // v_vec_size):
                    T.clear(grad_v_partial)
                    for i_h in T.unroll(hc_mult):
                        for i_k in T.vectorized(v_vec_size):
                            go_v_local[i_k] = go_smem[i_h, i * threads * v_vec_size + tid * v_vec_size + i_k]
                        for i_k in T.vectorized(v_vec_size):
                            grad_v_partial[i_k] += go_v_local[i_k] * gate_local[i_h]
                    for i_k in T.vectorized(v_vec_size):
                        grad_v[i_s, i * threads * v_vec_size + tid * v_vec_size + i_k] = grad_v_partial[i_k]

                dot_x_local = dot_in_local * rstd_x_local * rstd_x_local / hidden_size
                dot_k_local = dot_in_local * rstd_k_local * rstd_k_local / hidden_size

                # === Pass 2: grad_x, grad_k, grad_w — pipelined x/k, 2 warps per head ===
                x_sub_blks = x_blk_d // (threads_per_head * x_vec_size)
                num_x_tiles = hidden_size // x_blk_d

                for i_b in T.unroll(1, num_x_tiles):
                    phase = i_b % 2
                    prev = (i_b - 1) % 2

                    T.async_copy(hidden_states[i_s, :, i_b * x_blk_d:(i_b + 1) * x_blk_d],
                                 x_smem[phase, :, :], loop_layout=x_copy_layout)
                    T.async_copy(k[i_s, :, i_b * x_blk_d:(i_b + 1) * x_blk_d],
                                 k_smem[phase, :, :], loop_layout=x_copy_layout)
                    T.async_copy(weight_fused[:, i_b * x_blk_d:(i_b + 1) * x_blk_d],
                                 w_smem[phase, :, :], loop_layout=x_copy_layout)

                    T.ptx_wait_group(3 * x_sub_blks)

                    for i_sub in T.unroll(x_sub_blks):
                        sub_off = i_sub * (threads_per_head * x_vec_size) + sub_warp_id * (warp_size * x_vec_size)
                        global_base = (i_b - 1) * x_blk_d + sub_off
                        reg_base = ((i_b - 1) * x_sub_blks + i_sub) * x_vec_size
                        for i_k in T.vectorized(x_vec_size):
                            go_x_local[i_k] = go_smem[head_id, global_base + lane_id * x_vec_size + i_k]
                            x_local[i_k] = x_smem[prev, head_id, sub_off + lane_id * x_vec_size + i_k]
                            k_local[i_k] = k_smem[prev, head_id, sub_off + lane_id * x_vec_size + i_k]
                            w_fused_local[i_k] = w_smem[prev, head_id, sub_off + lane_id * x_vec_size + i_k]
                        for i_k in T.vectorized(x_vec_size):
                            grad_x[i_s, head_id, global_base + lane_id * x_vec_size + i_k] = \
                                go_x_local[i_k] + dldg_r[0] * (k_local[i_k] * w_fused_local[i_k] - x_local[i_k] * dot_x_local)
                            grad_k[i_s, head_id, global_base + lane_id * x_vec_size + i_k] = \
                                dldg_r[0] * (x_local[i_k] * w_fused_local[i_k] - k_local[i_k] * dot_k_local)
                        for i_k in T.serial(x_vec_size):
                            grad_w_local[reg_base + i_k] += dldg_r[0] * x_local[i_k] * k_local[i_k]

                # Epilogue: process last x/k/w tile + prefetch next token's go
                T.ptx_wait_group(0)

                # ensure v_smem is ready and go_smem[:go_blk_d] is clean
                T.sync_threads()
                if i_s + 1 < t_end:
                    T.async_copy(grad_out[i_s + 1, :, :go_blk_d], go_smem[:, :go_blk_d], loop_layout=go_copy_layout)

                for i_sub in T.unroll(x_sub_blks):
                    sub_off = i_sub * (threads_per_head * x_vec_size) + sub_warp_id * (warp_size * x_vec_size)
                    global_base = (num_x_tiles - 1) * x_blk_d + sub_off
                    reg_base = ((num_x_tiles - 1) * x_sub_blks + i_sub) * x_vec_size
                    for i_k in T.vectorized(x_vec_size):
                        go_x_local[i_k] = go_smem[head_id, global_base + lane_id * x_vec_size + i_k]
                        x_local[i_k] = x_smem[(num_x_tiles - 1) % 2, head_id, sub_off + lane_id * x_vec_size + i_k]
                        k_local[i_k] = k_smem[(num_x_tiles - 1) % 2, head_id, sub_off + lane_id * x_vec_size + i_k]
                        w_fused_local[i_k] = w_smem[(num_x_tiles - 1) % 2, head_id, sub_off + lane_id * x_vec_size + i_k]
                    for i_k in T.vectorized(x_vec_size):
                        grad_x[i_s, head_id, global_base + lane_id * x_vec_size + i_k] = \
                            go_x_local[i_k] + dldg_r[0] * (k_local[i_k] * w_fused_local[i_k] - x_local[i_k] * dot_x_local)
                        grad_k[i_s, head_id, global_base + lane_id * x_vec_size + i_k] = \
                            dldg_r[0] * (x_local[i_k] * w_fused_local[i_k] - k_local[i_k] * dot_k_local)
                    for i_k in T.serial(x_vec_size):
                        grad_w_local[reg_base + i_k] += dldg_r[0] * x_local[i_k] * k_local[i_k]

            # Write grad_w_local to global
            for i_reg in T.unroll(elems_per_warp_pair // x_vec_size):
                global_off = (i_reg * threads_per_head + sub_warp_id * warp_size + lane_id) * x_vec_size
                for i_k in T.vectorized(x_vec_size):
                    grad_w_partial[pid_p, head_id, global_off + i_k] = grad_w_local[i_reg * x_vec_size + i_k]

    return engram_gate_bwd_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
    },
)
def get_engram_gate_bwd_reload_kernel(
    hidden_size: int,
    scalar: float,
    k_stride_s: int,
    k_stride_h: int,
    v_stride_s: int,
    num_persistent_blocks: int,
    clamp_value: float = 1e-6,
    hc_mult: int = 4,
):
    """Low-LDS backward (reload path): Pass1/Pass2 slice streaming with ping-pong smem.

    Pass 1 uses ``h_blk_pass1`` tiles (dldg + grad_v). Pass 2 uses a smaller ``h_blk_pass2``
    (default 512 when valid) with ``x_vec_size=4`` and ping-pong ``go/x/k/w`` loads.
    """

    assert hc_mult == 4
    num_tokens = T.dynamic('num_tokens')
    warp_size = 64 if torch.version.hip is not None else 32
    warps_per_head = 1 if torch.version.hip is not None else 2
    num_warps = hc_mult * warps_per_head
    threads = warp_size * num_warps
    threads_per_head = warp_size * warps_per_head
    assert hidden_size % threads == 0
    elems_per_thread = hidden_size // threads
    elems_per_warp_pair = hidden_size // threads_per_head
    x_vec_size = 4

    v_vec_size = engram_gate_bwd_v_vec_size(elems_per_thread)
    go_vec_size = engram_gate_bwd_go_vec_size(hidden_size, threads_per_head)
    threads, h_blk1 = choose_reload_hidden_block(hidden_size, hc_mult)
    h_blk2 = choose_reload_pass2_hidden_block(hidden_size, hc_mult)
    x_vec_size2 = RELOAD_PASS2_X_VEC_SIZE
    num_slices1 = hidden_size // h_blk1
    num_slices2 = hidden_size // h_blk2
    go_sub_blks_slice = h_blk1 // (threads_per_head * go_vec_size)
    x_sub_blks_slice2 = h_blk2 // (threads_per_head * x_vec_size2)
    pass1_prefetch_wait_groups = reload_pass1_prefetch_wait_groups(
        h_blk1, threads_per_head, go_vec_size, threads, v_vec_size
    )
    pass2_prefetch_wait_groups = reload_pass2_prefetch_wait_groups(x_sub_blks_slice2)

    def smem_layout(i, j, vs):
        thread_id = i * threads_per_head + (j // vs) % threads_per_head
        local_id = (j % vs) + (j // (threads_per_head * vs)) * vs
        return thread_id, local_id

    go_copy_layout = T.Fragment((hc_mult, h_blk1), forward_fn=partial(smem_layout, vs=go_vec_size))
    go2_copy_layout = T.Fragment((hc_mult, h_blk2), forward_fn=partial(smem_layout, vs=x_vec_size2))
    x2_copy_layout = go2_copy_layout

    @T.prim_func
    def engram_gate_bwd_reload_kernel(
        grad_out: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        k: T.StridedTensor[(num_tokens, hc_mult, hidden_size), (k_stride_s, k_stride_h, 1), T.bfloat16],
        v: T.StridedTensor[(num_tokens, hidden_size), (v_stride_s, 1), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float],
        dot_in: T.Tensor[(num_tokens, hc_mult), T.float],
        gate_in: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_x_in: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_k_in: T.Tensor[(num_tokens, hc_mult), T.float],
        grad_x: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        grad_k: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        grad_v: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
        grad_w_partial: T.Tensor[(num_persistent_blocks, hc_mult, hidden_size), T.float],
    ):
        with T.Kernel(num_persistent_blocks, threads=threads) as pid_p:
            tid = T.get_thread_binding()
            warp_id = T.get_warp_idx()
            lane_id = T.get_lane_idx()
            head_id = warp_id // warps_per_head
            sub_warp_id = warp_id % warps_per_head

            go_local = T.alloc_local((go_vec_size,), T.float)
            v_local = T.alloc_local((go_vec_size,), T.float)
            go_v_local = T.alloc_local((v_vec_size,), T.float)
            go_x_local = T.alloc_local((x_vec_size,), T.float)
            x_local = T.alloc_local((x_vec_size,), T.float)
            k_local = T.alloc_local((x_vec_size,), T.float)
            w_fused_local = T.alloc_local((x_vec_size,), T.float)
            grad_v_partial = T.alloc_local((v_vec_size,), T.float)

            dldg_local = T.alloc_local((1,), T.float)
            dldg_r = T.alloc_local((1,), T.float)

            gate_local = T.alloc_local((hc_mult,), T.float)
            gate_local_hc = T.alloc_var(T.float)
            rstd_x_local = T.alloc_var(T.float)
            rstd_k_local = T.alloc_var(T.float)
            dot_in_local = T.alloc_var(T.float)
            dot_x_local = T.alloc_var(T.float)
            dot_k_local = T.alloc_var(T.float)

            go_bb = T.alloc_shared((2, hc_mult, h_blk1), T.bfloat16)
            v_bb = T.alloc_shared((2, h_blk1), T.bfloat16)
            go2_bb = T.alloc_shared((2, hc_mult, h_blk2), T.bfloat16)
            xh_bb = T.alloc_shared((2, hc_mult, h_blk2), T.bfloat16)
            kh_bb = T.alloc_shared((2, hc_mult, h_blk2), T.bfloat16)
            wf_bb = T.alloc_shared((2, hc_mult, h_blk2), T.float)
            dldg_smem = T.alloc_shared((hc_mult, warps_per_head), T.float)

            grad_w_local = T.alloc_local((elems_per_warp_pair,), T.float)

            i_grp_per_slice = h_blk1 // (threads * v_vec_size)

            per_block = T.ceildiv(num_tokens, num_persistent_blocks)
            t_start = T.min(per_block * pid_p, num_tokens)
            t_end = T.min(per_block * (pid_p + 1), num_tokens)

            T.clear(grad_w_local)

            for i_s in T.serial(t_start, t_end):
                T.copy(gate_in[i_s, :], gate_local)
                gate_local_hc = gate_in[i_s, head_id]
                rstd_x_local = rstd_x_in[i_s, head_id]
                rstd_k_local = rstd_k_in[i_s, head_id]
                dot_in_local = dot_in[i_s, head_id]

                T.clear(dldg_local)

                # Pass 1: ping-pong grad_out + v — dldg and grad_v per slice
                # prologue
                T.async_copy(grad_out[i_s, :, 0:h_blk1], go_bb[0, :, :], loop_layout=go_copy_layout)
                T.async_copy(v[i_s, 0:h_blk1], v_bb[0, :])

                for i_sl in T.serial(1, num_slices1):
                    phase = i_sl % 2
                    slice_lo_next = i_sl * h_blk1
                    T.async_copy(
                        grad_out[i_s, :, slice_lo_next : slice_lo_next + h_blk1],
                        go_bb[phase, :, :],
                        loop_layout=go_copy_layout,
                    )
                    T.async_copy(v[i_s, slice_lo_next : slice_lo_next + h_blk1], v_bb[phase, :])
                    T.ptx_wait_group(pass1_prefetch_wait_groups)
                    T.sync_threads()

                    prev = (i_sl - 1) % 2
                    slice_lo = (i_sl - 1) * h_blk1
                    for i_sub in T.serial(go_sub_blks_slice):
                        go_base_rel = (
                            i_sub * threads_per_head * go_vec_size
                            + sub_warp_id * warp_size * go_vec_size
                        )
                        for i_k in T.vectorized(go_vec_size):
                            go_local[i_k] = go_bb[prev, head_id, go_base_rel + lane_id * go_vec_size + i_k]
                            v_local[i_k] = v_bb[prev, go_base_rel + lane_id * go_vec_size + i_k]
                        for i_k in T.serial(go_vec_size):
                            dldg_local[0] += go_local[i_k] * v_local[i_k]

                    for jj in T.serial(i_grp_per_slice):
                        T.clear(grad_v_partial)
                        for i_h in T.unroll(hc_mult):
                            for i_k in T.vectorized(v_vec_size):
                                flat = jj * threads * v_vec_size + tid * v_vec_size + i_k
                                go_v_local[i_k] = go_bb[prev, i_h, flat]
                            for i_k in T.vectorized(v_vec_size):
                                grad_v_partial[i_k] += go_v_local[i_k] * gate_local[i_h]
                        for i_k in T.vectorized(v_vec_size):
                            gidx = slice_lo + jj * threads * v_vec_size + tid * v_vec_size + i_k
                            grad_v[i_s, gidx] = grad_v_partial[i_k]
                    T.sync_warp()

                T.ptx_wait_group(0)
                T.sync_threads()
                # Pass 2: ping-pong grad_out + x/k/w for grad_x/k/w, prefetch
                T.async_copy(grad_out[i_s, :, 0:h_blk2], go2_bb[0, :, :], loop_layout=go2_copy_layout)
                T.async_copy(hidden_states[i_s, :, 0:h_blk2], xh_bb[0, :, :], loop_layout=x2_copy_layout)
                T.async_copy(k[i_s, :, 0:h_blk2], kh_bb[0, :, :], loop_layout=x2_copy_layout)
                T.async_copy(weight_fused[:, 0:h_blk2], wf_bb[0, :, :], loop_layout=x2_copy_layout)

                # Pass 1 last prologue
                last_phase1 = (num_slices1 - 1) % 2
                slice_lo1 = (num_slices1 - 1) * h_blk1
                for i_sub in T.serial(go_sub_blks_slice):
                    go_base_rel = (
                        i_sub * threads_per_head * go_vec_size + sub_warp_id * warp_size * go_vec_size
                    )
                    for i_k in T.vectorized(go_vec_size):
                        go_local[i_k] = go_bb[last_phase1, head_id, go_base_rel + lane_id * go_vec_size + i_k]
                        v_local[i_k] = v_bb[last_phase1, go_base_rel + lane_id * go_vec_size + i_k]
                    for i_k in T.serial(go_vec_size):
                        dldg_local[0] += go_local[i_k] * v_local[i_k]

                for jj in T.serial(i_grp_per_slice):
                    T.clear(grad_v_partial)
                    for i_h in T.unroll(hc_mult):
                        for i_k in T.vectorized(v_vec_size):
                            flat = jj * threads * v_vec_size + tid * v_vec_size + i_k
                            go_v_local[i_k] = go_bb[last_phase1, i_h, flat]
                        for i_k in T.vectorized(v_vec_size):
                            grad_v_partial[i_k] += go_v_local[i_k] * gate_local[i_h]
                    for i_k in T.vectorized(v_vec_size):
                        gidx = slice_lo1 + jj * threads * v_vec_size + tid * v_vec_size + i_k
                        grad_v[i_s, gidx] = grad_v_partial[i_k]

                dldg_local[0] = T.warp_reduce_sum(dldg_local[0])
                if lane_id == 0:
                    dldg_smem[head_id, sub_warp_id] = dldg_local[0]
                T.sync_threads()

                T.clear(dldg_r)
                for i_w in T.unroll(warps_per_head):
                    dldg_r[0] += dldg_smem[head_id, i_w]
                dldg_r[0] = T.Select(
                    T.abs(dot_in_local) * scalar * rstd_x_local * rstd_k_local < clamp_value,
                    0.0,
                    dldg_r[0]
                    * gate_local_hc
                    * (1.0 - gate_local_hc)
                    * 0.5
                    * T.sqrt(scalar * rstd_x_local * rstd_k_local / T.abs(dot_in_local)),
                )

                dot_x_local = dot_in_local * rstd_x_local * rstd_x_local / hidden_size
                dot_k_local = dot_in_local * rstd_k_local * rstd_k_local / hidden_size

                for i_sl in T.serial(1, num_slices2):
                    phase = i_sl % 2
                    slice_lo_next = i_sl * h_blk2
                    T.async_copy(
                        grad_out[i_s, :, slice_lo_next : slice_lo_next + h_blk2],
                        go2_bb[phase, :, :],
                        loop_layout=go2_copy_layout,
                    )
                    T.async_copy(
                        hidden_states[i_s, :, slice_lo_next : slice_lo_next + h_blk2],
                        xh_bb[phase, :, :],
                        loop_layout=x2_copy_layout,
                    )
                    T.async_copy(
                        k[i_s, :, slice_lo_next : slice_lo_next + h_blk2],
                        kh_bb[phase, :, :],
                        loop_layout=x2_copy_layout,
                    )
                    T.async_copy(
                        weight_fused[:, slice_lo_next : slice_lo_next + h_blk2],
                        wf_bb[phase, :, :],
                        loop_layout=x2_copy_layout,
                    )
                    T.ptx_wait_group(pass2_prefetch_wait_groups)
                    # T.sync_threads()

                    prev = (i_sl - 1) % 2
                    slice_lo = (i_sl - 1) * h_blk2
                    for i_sub in T.serial(x_sub_blks_slice2):
                        sub_off = (
                            i_sub * (threads_per_head * x_vec_size2)
                            + sub_warp_id * (warp_size * x_vec_size2)
                        )
                        global_base = slice_lo + sub_off
                        reg_base = ((i_sl - 1) * x_sub_blks_slice2 + i_sub) * x_vec_size2
                        for i_k in T.vectorized(x_vec_size2):
                            go_x_local[i_k] = go2_bb[prev, head_id, sub_off + lane_id * x_vec_size2 + i_k]
                            x_local[i_k] = xh_bb[prev, head_id, sub_off + lane_id * x_vec_size2 + i_k]
                            k_local[i_k] = kh_bb[prev, head_id, sub_off + lane_id * x_vec_size2 + i_k]
                            w_fused_local[i_k] = wf_bb[prev, head_id, sub_off + lane_id * x_vec_size2 + i_k]
                        for i_k in T.vectorized(x_vec_size2):
                            grad_x[i_s, head_id, global_base + lane_id * x_vec_size2 + i_k] = (
                                go_x_local[i_k]
                                + dldg_r[0]
                                * (k_local[i_k] * w_fused_local[i_k] - x_local[i_k] * dot_x_local)
                            )
                            grad_k[i_s, head_id, global_base + lane_id * x_vec_size2 + i_k] = dldg_r[0] * (
                                x_local[i_k] * w_fused_local[i_k] - k_local[i_k] * dot_k_local
                            )
                        for i_k in T.serial(x_vec_size2):
                            grad_w_local[reg_base + i_k] += dldg_r[0] * x_local[i_k] * k_local[i_k]

                T.ptx_wait_group(0)
                # T.sync_threads()

                last_phase2 = (num_slices2 - 1) % 2
                slice_lo2 = (num_slices2 - 1) * h_blk2
                for i_sub in T.serial(x_sub_blks_slice2):
                    sub_off = (
                        i_sub * (threads_per_head * x_vec_size2)
                        + sub_warp_id * (warp_size * x_vec_size2)
                    )
                    global_base = slice_lo2 + sub_off
                    reg_base = ((num_slices2 - 1) * x_sub_blks_slice2 + i_sub) * x_vec_size2
                    for i_k in T.vectorized(x_vec_size2):
                        go_x_local[i_k] = go2_bb[last_phase2, head_id, sub_off + lane_id * x_vec_size2 + i_k]
                        x_local[i_k] = xh_bb[last_phase2, head_id, sub_off + lane_id * x_vec_size2 + i_k]
                        k_local[i_k] = kh_bb[last_phase2, head_id, sub_off + lane_id * x_vec_size2 + i_k]
                        w_fused_local[i_k] = wf_bb[last_phase2, head_id, sub_off + lane_id * x_vec_size2 + i_k]
                    for i_k in T.vectorized(x_vec_size2):
                        grad_x[i_s, head_id, global_base + lane_id * x_vec_size2 + i_k] = (
                            go_x_local[i_k]
                            + dldg_r[0]
                            * (k_local[i_k] * w_fused_local[i_k] - x_local[i_k] * dot_x_local)
                        )
                        grad_k[i_s, head_id, global_base + lane_id * x_vec_size2 + i_k] = dldg_r[0] * (
                            x_local[i_k] * w_fused_local[i_k] - k_local[i_k] * dot_k_local
                        )
                    for i_k in T.serial(x_vec_size2):
                        grad_w_local[reg_base + i_k] += dldg_r[0] * x_local[i_k] * k_local[i_k]

            for i_reg in T.unroll(elems_per_warp_pair // x_vec_size):
                global_off = (i_reg * threads_per_head + sub_warp_id * warp_size + lane_id) * x_vec_size
                for i_k in T.vectorized(x_vec_size):
                    grad_w_partial[pid_p, head_id, global_off + i_k] = grad_w_local[i_reg * x_vec_size + i_k]

    return engram_gate_bwd_reload_kernel


def engram_gate_fwd(
    hidden_states: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight_fused: torch.Tensor,
    eps: float,
    clamp_value: float,
    save_for_backward: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Engram gate forward pass.

    Args:
        hidden_states: Input of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        k: Key embeddings of shape (num_tokens, hc_mult, hidden_size), bfloat16.
        v: Value embeddings of shape (num_tokens, hidden_size), bfloat16.
        weight_fused: Fused RMSNorm weight (weight_hidden * weight_embed), shape (hc_mult, hidden_size), float32.
        eps: Epsilon for RMSNorm numerical stability.
        clamp_value: Clamp threshold for signed-sqrt gate activation.
        save_for_backward: If True, saves dot/gate_score/rstd_x/rstd_k for backward. If False,
            returns None for those intermediates (inference mode).

    Returns:
        tuple: (output, dot, gate_score, rstd_x, rstd_k). When save_for_backward=False,
            dot/gate_score/rstd_x/rstd_k are None.
    """
    num_tokens, hc_mult, hidden_size = hidden_states.shape
    scalar = hidden_size**-0.5
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1
    k_stride_s, k_stride_h, _ = k.stride()
    v_stride_s, _ = v.stride()

    kernel = get_engram_gate_fwd_kernel(hidden_size, eps, scalar, k_stride_s, k_stride_h, v_stride_s, get_num_sms(), clamp_value, hc_mult, save_for_backward)
    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    output = torch.empty_like(hidden_states)
    if save_for_backward:
        dot = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        gate_score = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        rstd_x = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        rstd_k = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
    else:
        # TileLang JIT adapter does not accept None tensor arguments.
        # Provide shape-compatible placeholders when save tensors are disabled.
        dot = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        gate_score = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        rstd_x = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
        rstd_k = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)

    kernel(hidden_states, k, v, weight_fused, output, dot, gate_score, rstd_x, rstd_k)

    if not save_for_backward:
        return output, None, None, None, None

    return output, dot, gate_score, rstd_x, rstd_k


def engram_gate_bwd(
    grad_out: torch.Tensor,
    hidden_states: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight_fused: torch.Tensor,
    dot: torch.Tensor,
    gate_score: torch.Tensor,
    rstd_x: torch.Tensor,
    rstd_k: torch.Tensor,
    clamp_value: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Engram gate backward: computes grad_hidden_states, grad_k, grad_v, grad_w_partial.

    Args:
        grad_out: Gradient of output, shape (num_tokens, hc_mult, hidden_size), bfloat16.
        hidden_states: Original input from forward, shape (num_tokens, hc_mult, hidden_size), bfloat16.
        k: Original key embeddings from forward, shape (num_tokens, hc_mult, hidden_size), bfloat16.
        v: Original value embeddings from forward, shape (num_tokens, hidden_size), bfloat16.
        weight_fused: Fused RMSNorm weight (weight_hidden * weight_embed), shape (hc_mult, hidden_size), float32.
        dot: Saved scaled dot product from forward, shape (num_tokens, hc_mult), float32.
        gate_score: Saved gate scores from forward, shape (num_tokens, hc_mult), float32.
        rstd_x: Saved reciprocal std of x from forward, shape (num_tokens, hc_mult), float32.
        rstd_k: Saved reciprocal std of k from forward, shape (num_tokens, hc_mult), float32.
        clamp_value: Clamp threshold for signed-sqrt gate activation.

    Returns:
        tuple: (grad_hidden_states, grad_k, grad_v, grad_w_partial) where grad_w_partial
            has shape (num_persistent_blocks, hc_mult, hidden_size) and needs further reduction.
    """
    num_tokens, hc_mult, hidden_size = hidden_states.shape
    scalar = hidden_size**-0.5
    assert k.stride(-1) == 1
    assert v.stride(-1) == 1
    k_stride_s, k_stride_h, _ = k.stride()
    v_stride_s, _ = v.stride()

    force_reload = os.environ.get('TK_ENGRAM_GATE_BWD_FORCE_FRAGMENT') in {'1', 'true', 'True'}
    use_reload = force_reload or engram_gate_bwd_reload_pipeline_exceeds_smem_budget(hidden_size, hc_mult)
    kw = dict(
        scalar=scalar,
        k_stride_s=k_stride_s,
        k_stride_h=k_stride_h,
        v_stride_s=v_stride_s,
        num_persistent_blocks=get_num_sms(),
        clamp_value=clamp_value,
        hc_mult=hc_mult,
    )
    kernel = (
        get_engram_gate_bwd_reload_kernel(hidden_size, **kw)
        if use_reload
        else get_engram_gate_bwd_kernel(hidden_size, **kw)
    )
    if int(os.getenv('TK_PRINT_KERNEL_SOURCE', 0)):
        print(kernel.get_kernel_source())

    grad_hidden_states = torch.empty_like(hidden_states)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    grad_w_partial = torch.empty((get_num_sms(), hc_mult, hidden_size), dtype=torch.float32, device=hidden_states.device)

    kernel(grad_out, hidden_states, k, v, weight_fused,
           dot, gate_score, rstd_x, rstd_k,
           grad_hidden_states, grad_k, grad_v, grad_w_partial)

    return grad_hidden_states, grad_k, grad_v, grad_w_partial

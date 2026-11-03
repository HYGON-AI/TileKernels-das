# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# SPDX-License-Identifier: MIT

"""DAS low-LDS Engram backward, adapted to upstream packed-kv and mask APIs.

Arithmetic, ping-pong buffers, async wait counts and wave synchronization are
preserved from the pre-merge DAS reload kernel (7ee2019).
"""

from functools import partial

import torch
import tilelang
from tilelang import language as T

from tile_kernels.engram.engram_gate_bwd_reload_helpers import (
    RELOAD_PASS2_X_VEC_SIZE,
    choose_reload_hidden_block,
    choose_reload_pass2_hidden_block,
    engram_gate_bwd_go_vec_size,
    engram_gate_bwd_v_vec_size,
    reload_pass1_prefetch_wait_groups,
    reload_pass2_prefetch_wait_groups,
)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_OUT_OF_BOUND_WARNING: True,
    },
)
def get_engram_gate_bwd_reload_kernel_hcu(
    hidden_size: int,
    scalar: float,
    num_persistent_blocks: int,
    clamp_value: float = 1e-6,
    hc_mult: int = 4,
    has_image_token_mask: bool = False,
    use_pdl: bool = False,
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
        kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float],
        dot_in: T.Tensor[(num_tokens, hc_mult), T.float],
        gate_in: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_x_in: T.Tensor[(num_tokens, hc_mult), T.float],
        rstd_k_in: T.Tensor[(num_tokens, hc_mult), T.float],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        grad_x: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        grad_kv: T.Tensor[(num_tokens, hc_mult + 1, hidden_size), T.bfloat16],
        grad_w_partial: T.Tensor[(num_persistent_blocks, hc_mult, hidden_size), T.float],
    ):
        with T.Kernel(num_persistent_blocks, threads=threads) as pid_p:
            if use_pdl:
                T.pdl_sync()
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
                if has_image_token_mask and image_token_mask[i_s]:
                    for i_h, i_d in T.Parallel(hc_mult, hidden_size):
                        grad_x[i_s, i_h, i_d] = grad_out[i_s, i_h, i_d]
                    for i_h, i_d in T.Parallel(hc_mult + 1, hidden_size):
                        grad_kv[i_s, i_h, i_d] = 0.0
                    continue
                T.copy(gate_in[i_s, :], gate_local)
                gate_local_hc = gate_in[i_s, head_id]
                rstd_x_local = rstd_x_in[i_s, head_id]
                rstd_k_local = rstd_k_in[i_s, head_id]
                dot_in_local = dot_in[i_s, head_id]

                T.clear(dldg_local)

                # Pass 1: ping-pong grad_out + v — dldg and grad_v per slice
                # prologue
                T.async_copy(grad_out[i_s, :, 0:h_blk1], go_bb[0, :, :], loop_layout=go_copy_layout)
                T.async_copy(kv[i_s, hc_mult, 0:h_blk1], v_bb[0, :])

                for i_sl in T.serial(1, num_slices1):
                    phase = i_sl % 2
                    slice_lo_next = i_sl * h_blk1
                    T.async_copy(
                        grad_out[i_s, :, slice_lo_next : slice_lo_next + h_blk1],
                        go_bb[phase, :, :],
                        loop_layout=go_copy_layout,
                    )
                    T.async_copy(kv[i_s, hc_mult, slice_lo_next : slice_lo_next + h_blk1], v_bb[phase, :])
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
                            grad_kv[i_s, hc_mult, gidx] = grad_v_partial[i_k]
                    T.sync_warp()

                T.ptx_wait_group(0)
                T.sync_threads()
                # Pass 2: ping-pong grad_out + x/k/w for grad_x/k/w, prefetch
                T.async_copy(grad_out[i_s, :, 0:h_blk2], go2_bb[0, :, :], loop_layout=go2_copy_layout)
                T.async_copy(hidden_states[i_s, :, 0:h_blk2], xh_bb[0, :, :], loop_layout=x2_copy_layout)
                T.async_copy(kv[i_s, :hc_mult, 0:h_blk2], kh_bb[0, :, :], loop_layout=x2_copy_layout)
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
                        grad_kv[i_s, hc_mult, gidx] = grad_v_partial[i_k]

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
                        kv[i_s, :hc_mult, slice_lo_next : slice_lo_next + h_blk2],
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
                            grad_kv[i_s, head_id, global_base + lane_id * x_vec_size2 + i_k] = dldg_r[0] * (
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
                        grad_kv[i_s, head_id, global_base + lane_id * x_vec_size2 + i_k] = dldg_r[0] * (
                            x_local[i_k] * w_fused_local[i_k] - k_local[i_k] * dot_k_local
                        )
                    for i_k in T.serial(x_vec_size2):
                        grad_w_local[reg_base + i_k] += dldg_r[0] * x_local[i_k] * k_local[i_k]

            for i_reg in T.unroll(elems_per_warp_pair // x_vec_size):
                global_off = (i_reg * threads_per_head + sub_warp_id * warp_size + lane_id) * x_vec_size
                for i_k in T.vectorized(x_vec_size):
                    grad_w_partial[pid_p, head_id, global_off + i_k] = grad_w_local[i_reg * x_vec_size + i_k]

            if use_pdl:
                T.pdl_trigger()

    return engram_gate_bwd_reload_kernel

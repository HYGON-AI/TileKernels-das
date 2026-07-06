# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# SPDX-License-Identifier: MIT

import tilelang
import torch
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_split_sinkhorn_fwd_orig(
    hc: int,
    sinkhorn_iters: int,
    eps: float,
    threads: int,
) -> tilelang.JITKernel:
    n = T.dynamic('n')
    mix_hc = (2 + hc) * hc

    @T.prim_func
    def _mhc_split_sinkhorn_fwd_orig_kernel(
        mixes: T.Tensor[(n, mix_hc), T.float32],
        hc_scale: T.Tensor[(3,), T.float32],
        hc_base: T.Tensor[(mix_hc,), T.float32],
        pre: T.Tensor[(n, hc), T.float32],
        post: T.Tensor[(n, hc), T.float32],
        comb: T.Tensor[(n, hc, hc), T.float32],
    ) -> None:
        with T.Kernel(n, threads=threads) as i:
            mixes_shared = T.alloc_shared(mix_hc, T.float32)
            comb_frag = T.alloc_fragment((hc, hc), T.float32)
            row_sum = T.alloc_fragment(hc, T.float32)
            col_sum = T.alloc_fragment(hc, T.float32)
            row_max = T.alloc_fragment(hc, T.float32)

            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = (
                    mixes_shared[j * hc + k + hc * 2] * hc_scale[2]
                    + hc_base[j * hc + k + hc * 2]
                )

            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])

    return _mhc_split_sinkhorn_fwd_orig_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_split_sinkhorn_fwd(
    hc: int,
    sinkhorn_iters: int,
    eps: float,
    token_block_size: int,
    threads: int,
) -> tilelang.JITKernel:
    n = T.dynamic('n')
    mix_hc = (2 + hc) * hc
    @T.prim_func
    def _mhc_split_sinkhorn_fwd_kernel(
        mixes: T.Tensor[(n, mix_hc), T.float32],
        hc_scale: T.Tensor[(3,), T.float32],
        hc_base: T.Tensor[(mix_hc,), T.float32],
        pre: T.Tensor[(n, hc), T.float32],
        post: T.Tensor[(n, hc), T.float32],
        comb: T.Tensor[(n, hc, hc), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(n, token_block_size), threads=threads) as pid_x:
            mixes_shared = T.alloc_shared((token_block_size, mix_hc), T.float32)
            comb_frag = T.alloc_fragment((token_block_size, hc, hc), T.float32)
            row_sum = T.alloc_fragment((token_block_size, hc), T.float32)
            col_sum = T.alloc_fragment((token_block_size, hc), T.float32)
            row_max = T.alloc_fragment((token_block_size, hc), T.float32)
            T.copy(mixes[pid_x * token_block_size, 0], mixes_shared)
            for i, j in T.Parallel(token_block_size, hc):
                idx = pid_x * token_block_size + i
                if idx < n:
                    pre[idx, j] = T.sigmoid(mixes_shared[i, j] * hc_scale[0] + hc_base[j]) + eps
            for i, j in T.Parallel(token_block_size, hc):
                idx = pid_x * token_block_size + i
                if idx < n:
                    post[idx, j] = 2 * T.sigmoid(
                        mixes_shared[i, j + hc] * hc_scale[1] + hc_base[j + hc]
                    )

            for i, j, k in T.Parallel(token_block_size, hc, hc):
                comb_frag[i, j, k] = (
                    mixes_shared[i, j * hc + k + hc * 2] * hc_scale[2]
                    + hc_base[j * hc + k + hc * 2]
                )

            T.reduce_max(comb_frag, row_max, dim=2)
            for i, j, k in T.Parallel(token_block_size, hc, hc):
                comb_frag[i, j, k] = T.exp(comb_frag[i, j, k] - row_max[i, j])
            T.reduce_sum(comb_frag, row_sum, dim=2)
            for i, j, k in T.Parallel(token_block_size, hc, hc):
                comb_frag[i, j, k] = comb_frag[i, j, k] / row_sum[i, j] + eps

            T.reduce_sum(comb_frag, col_sum, dim=1)
            for i, j, k in T.Parallel(token_block_size, hc, hc):
                comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                T.reduce_sum(comb_frag, row_sum, dim=2)
                for i, j, k in T.Parallel(token_block_size, hc, hc):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / (row_sum[i, j] + eps)
                T.reduce_sum(comb_frag, col_sum, dim=1)
                for i, j, k in T.Parallel(token_block_size, hc, hc):
                    comb_frag[i, j, k] = comb_frag[i, j, k] / (col_sum[i, k] + eps)

            for i, j, k in T.Parallel(token_block_size, hc, hc):
                idx = pid_x * token_block_size + i
                if idx < n:
                    comb[idx, j, k] = comb_frag[i, j, k]

    return _mhc_split_sinkhorn_fwd_kernel


def mhc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    token_block_size: int = 32,
    threads: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, _ = mixes.size()
    n = b * s
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    if threads * token_block_size // 4 > n:
        kernel = _mhc_split_sinkhorn_fwd_orig(hc_mult, sinkhorn_iters, eps, threads)
    else:
        kernel = _mhc_split_sinkhorn_fwd(hc_mult, sinkhorn_iters, eps, token_block_size, threads)
    kernel(
        mixes.contiguous().view(-1, (2 + hc_mult) * hc_mult),
        hc_scale.contiguous(),
        hc_base.contiguous(),
        pre.view(-1, hc_mult),
        post.view(-1, hc_mult),
        comb.view(-1, hc_mult, hc_mult),
    )
    return pre, post, comb

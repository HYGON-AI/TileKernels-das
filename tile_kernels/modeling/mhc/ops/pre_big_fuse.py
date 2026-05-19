import functools
import math
from typing import NamedTuple

import tilelang
import torch
from tilelang import language as T

from tile_kernels.mhc.norm_fn_kernel import _mhc_pre_norm_fn_fwd_mul
from tile_kernels.mhc.pre_norm_fn_splitk_kernel import mhc_pre_gemm_sqrsum_splitk_kernel
from tile_kernels.mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse

# Global guards for validating split-k stage0/stage1 kernels.
cu_count = torch.cuda.get_device_properties("cuda").multi_processor_count


class PreBigFuseBlockInfo(NamedTuple):
    token_block: int
    hidden_block: int
    hidden_loop: int
    n_splits_pre: int
    use_small_token_splitk: bool


@functools.lru_cache(maxsize=1024)
def get_block_info(num_tokens: int, mhc_hidden_size: int, cu_count: int) -> PreBigFuseBlockInfo:
    token_block = 128  # use 128 for better performance
    hidden_block = 128  # with hidden_block = 128, the occupancy is 2
    hidden_loop = mhc_hidden_size // hidden_block
    token_loop = (num_tokens + token_block - 1) // token_block

    if token_loop <= 2:
        if num_tokens > 128:
            # for occupied 2
            n_splits_pre = 64
            if hidden_loop % n_splits_pre != 0:
                hidden_block = 64
                hidden_loop = mhc_hidden_size // hidden_block
        elif num_tokens > 64:
            # for occupied 2
            token_block = 64
            n_splits_pre = 64
            if hidden_loop % n_splits_pre != 0:
                hidden_block = 64
                hidden_loop = mhc_hidden_size // hidden_block
        elif num_tokens > 32:
            # for occupied 2
            token_block = 32
            n_splits_pre = 64
            if hidden_loop % n_splits_pre != 0:
                hidden_block = 64
                hidden_loop = mhc_hidden_size // hidden_block
        else:
            # occupied 1
            token_block = 32
            n_splits_pre = 64
            if hidden_loop % n_splits_pre != 0:
                hidden_block = 64
                hidden_loop = mhc_hidden_size // hidden_block
    elif token_loop <= 4:
        n_splits_pre = 32
    elif token_loop <= cu_count // 8:
        n_splits_pre = 16
    elif token_loop <= cu_count // 4:
        n_splits_pre = 8
    elif token_loop <= cu_count * 0.75:
        n_splits_pre = 8
    elif token_loop <= cu_count * 2:
        n_splits_pre = 4
    else:
        n_splits_pre = 1

    final_token_loop = (num_tokens + token_block - 1) // token_block
    use_small_token_splitk = (
        n_splits_pre > 1
        and final_token_loop <= cu_count * 2
        and hidden_loop > 0
        and hidden_loop % n_splits_pre == 0
    )

    if not use_small_token_splitk:
        token_block = 64
        hidden_block = 128
    # print(f"use_small_token_splitk={use_small_token_splitk}, num_tokens={num_tokens}, hidden_loop={hidden_loop}, "
    #       f"MHC_PRE_BIG_FUSE_N_SPLITS_PRE={MHC_PRE_BIG_FUSE_N_SPLITS_PRE}, token_block={token_block}, hidden_block={hidden_block}")

    return PreBigFuseBlockInfo(
        token_block=token_block,
        hidden_block=hidden_block,
        hidden_loop=hidden_loop,
        n_splits_pre=n_splits_pre,
        use_small_token_splitk=use_small_token_splitk,
    )


@functools.lru_cache(maxsize=128)
def _round_to_tf32_kernel(n_elem: int) -> tilelang.JITKernel:
    return _compile_round_to_tf32(n_elem)


@tilelang.jit  # inp, out both passed in; out_idx would mean only inp is passed and out is allocated inside the adapter
def _compile_round_to_tf32(n_elem: int) -> tilelang.JITKernel:
    """Bitcast float32 -> int32, add 0x1000, bitcast back (1D linear scan for coalescing)."""
    _TF32_ROUND_BITS = 0x1000
    _ROUND_TO_TF32_BLK_MAX = 2048
    n_blk = math.gcd(_ROUND_TO_TF32_BLK_MAX, n_elem)

    @T.prim_func
    def _round_to_tf32_prim(
        inp: T.Tensor[(n_elem,), T.float32],
        out: T.Tensor[(n_elem,), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(n_elem, n_blk)) as pid:
            input_frag = T.alloc_fragment((n_blk,), T.float32)
            output_frag = T.alloc_fragment((n_blk,), T.float32)
            T.copy(inp[pid * n_blk], input_frag)
            input_int = T.view(input_frag, (n_blk,), T.int32)

            for t in T.Parallel(n_blk):
                input_int[t] += T.int32(_TF32_ROUND_BITS)
                output_frag[t] = T.reinterpret(input_int[t], T.float32)
            T.copy(output_frag, out[pid * n_blk])

    return _round_to_tf32_prim


def round_to_tf32(fn: torch.Tensor) -> torch.Tensor:
    """TF32 grid rounding via TileLang (flat numel; preserves original shape)."""
    ne = int(fn.numel())
    out = torch.empty_like(fn)
    _round_to_tf32_kernel(ne)(fn.reshape(ne), out.reshape(ne))
    return out


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    mhc_hidden_size = mhc_mult * hidden_size
    assert fn.shape[0] == mhc_mult3
    assert fn.shape[1] == mhc_hidden_size
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    post_mix = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty(num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device)

    # Bucket by 32 so get_block_info cache keys align with common launch granularity; real buffers still use num_tokens.
    num_tokens_align = (int(num_tokens) + 31) // 32 * 32
    block_info = get_block_info(num_tokens_align, mhc_hidden_size, cu_count)
    token_block = block_info.token_block
    hidden_block = block_info.hidden_block
    hidden_loop = block_info.hidden_loop
    MHC_PRE_BIG_FUSE_N_SPLITS_PRE = block_info.n_splits_pre
    use_small_token_splitk = block_info.use_small_token_splitk

    fn = round_to_tf32(fn)

    if use_small_token_splitk:
        kernel_0, kernel_1 = mhc_pre_gemm_sqrsum_splitk_kernel(
            mhc_mult3,
            mhc_hidden_size,
            split_k=MHC_PRE_BIG_FUSE_N_SPLITS_PRE,
            token_block=token_block,
            hidden_block=hidden_block,
        )
        partial_out = torch.empty(
            MHC_PRE_BIG_FUSE_N_SPLITS_PRE, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device
        )
        partial_sqrsum = torch.empty(
            MHC_PRE_BIG_FUSE_N_SPLITS_PRE, num_tokens, dtype=torch.float32, device=residual.device
        )
        # gemm_out_mul = torch.empty(
        #     1, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device
        # )
        # gemm_out_sqrsum = torch.empty(1, num_tokens, dtype=torch.float32, device=residual.device)
        kernel_0(
            residual_flat.view(-1, mhc_hidden_size),
            fn,
            partial_out,
            partial_sqrsum,
        )
        gemm_out_mul = partial_out
        gemm_out_sqrsum = partial_sqrsum
        # kernel_1(
        #     partial_out,
        #     partial_sqrsum,
        #     gemm_out_mul.squeeze(0),
        #     gemm_out_sqrsum.squeeze(0),
        # )
        n_splits = MHC_PRE_BIG_FUSE_N_SPLITS_PRE
    else:
        gemm_out_mul = torch.empty(
            1, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device
        )
        gemm_out_sqrsum = torch.empty(1, num_tokens, dtype=torch.float32, device=residual.device)
        n_splits = 1
        fwd_mul_kernel = _mhc_pre_norm_fn_fwd_mul(mhc_mult3, 1, mhc_hidden_size, token_block=token_block, hidden_block=hidden_block)
        fwd_mul_kernel(
            residual_flat.view(-1, mhc_hidden_size),
            fn,
            gemm_out_mul.view(-1, 1, mhc_mult3),
            gemm_out_sqrsum.view(-1, 1),
        )
    # END of TileLang implementation of pre-norm-fn forward matmul

    _mhc_pre_big_fuse(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input

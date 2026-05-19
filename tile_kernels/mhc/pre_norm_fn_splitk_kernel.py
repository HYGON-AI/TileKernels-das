import functools
from typing import Tuple

import tilelang
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}

@functools.cache
def mhc_pre_gemm_sqrsum_splitk_kernel(
    mhc_mult3: int,
    mhc_hidden_size: int,
    split_k: int,
    token_block: int = 64,
    hidden_block: int = 256,
    threads: int = 256,
) -> Tuple[tilelang.JITKernel, tilelang.JITKernel]:
    assert mhc_mult3 <= 32
    assert mhc_hidden_size % hidden_block == 0
    assert mhc_hidden_size % split_k == 0
    split_size = mhc_hidden_size // split_k
    assert split_size % hidden_block == 0

    num_tokens = T.dynamic("num_tokens")

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def mhc_pre_gemm_sqrsum_splitk_stage_0(
        x: T.Tensor[(num_tokens, mhc_hidden_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, mhc_hidden_size), T.float32],
        out_partial: T.Tensor[(split_k, num_tokens, mhc_mult3), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
    ):
        with T.Kernel(split_k, T.ceildiv(num_tokens, token_block), threads=threads) as (
            bz,
            px,
        ):
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sq_part4 = T.alloc_fragment((token_block, 16), T.float32)
            T.clear(out_frag)
            T.clear(sq_part4)

            k_base = bz * split_size

            for pz in T.Pipelined(split_size // hidden_block, num_stages=0):
                x_frag_pre = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
                fn_frag_pre = T.alloc_fragment((32, hidden_block), T.float32)
                x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
                x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
                fn_frag = T.alloc_fragment((32, hidden_block), T.float32)

                x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
                fn_smem = T.alloc_shared((32, hidden_block), T.float32)
                T.annotate_layout({x_smem_16: tilelang.layout.make_hcu_swizzled_layout(x_smem_16, major_pack=2)})
                T.annotate_layout({fn_smem: tilelang.layout.make_hcu_swizzled_layout(fn_smem, major_pack=2)})

                T.copy(x[px * token_block, k_base + pz * hidden_block], x_frag_pre)
                T.copy(fn[0, k_base + pz * hidden_block], fn_frag_pre)

                T.copy(x_frag_pre, x_smem_16)
                T.copy(x_smem_16, x_frag_16)
                T.copy(x_frag_16, x_frag)
                T.copy(fn_frag_pre, fn_smem)
                T.copy(fn_smem, fn_frag)
                for jj in T.serial(hidden_block // 16):
                    for i, j in T.Parallel(token_block, 16):
                        v = x_frag[i, jj * 16 + j]
                        sq_part4[i, j] += v * v

                T.gemm(
                    x_frag,
                    fn_frag,
                    out_frag,
                    transpose_A=False,
                    transpose_B=True,
                    k_pack=2,
                    policy=T.GemmWarpPolicy.FullRow,
                    use_tf32=True,
                )

            sq_l = T.alloc_fragment((token_block,), T.float32)
            T.reduce_sum(sq_part4, sq_l)
            out_shared = T.alloc_shared((token_block, 32), T.float32)
            T.annotate_layout({out_shared: tilelang.layout.make_hcu_swizzled_layout(out_shared, major_pack=2)})
            T.copy(out_frag, out_shared)

            for i in T.Parallel(token_block):
                t = px * token_block + i
                if t < num_tokens:
                    sqrsum_partial[bz, t] = sq_l[i]

            for i, j in T.Parallel(token_block, 32):
                t = px * token_block + i
                if t < num_tokens and j < mhc_mult3:
                    out_partial[bz, t, j] = out_shared[i, j]

    @tilelang.jit
    def mhc_pre_gemm_sqrsum_splitk_stage_1(
        out_partial: T.Tensor[(split_k, num_tokens, 32), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
        out: T.Tensor[(num_tokens, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens,), T.float32],
    ):
        warps_per_cta = threads // 64
        num_reduce = T.ceildiv(split_k, 64)
        with T.Kernel(T.ceildiv(num_tokens, warps_per_cta), threads=threads) as (px,):
            tx = T.get_thread_binding()
            warp = tx // 64
            lane = tx % 64
            t = px * warps_per_cta + warp
            s = T.alloc_local((1,), T.float32)
            acc = T.alloc_local((1,), T.float32)
            s[0] = 0
            acc[0] = 0

            if t < num_tokens:
                for r in T.serial(num_reduce):
                    bz = r * 64 + lane
                    s[0] += T.if_then_else(bz < split_k, sqrsum_partial[bz, t], 0.0)
                sqrsum[t] = T.warp_reduce_sum(s[0])
                if lane < mhc_mult3:
                    for bz in T.serial(split_k):
                        acc[0] += out_partial[bz, t, lane]
                    out[t, lane] = acc[0]

    return (
        mhc_pre_gemm_sqrsum_splitk_stage_0,
        mhc_pre_gemm_sqrsum_splitk_stage_1,
    )


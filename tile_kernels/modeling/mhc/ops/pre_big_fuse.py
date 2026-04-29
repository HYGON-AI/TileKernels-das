import torch

from tile_kernels.mhc.norm_fn_kernel import _mhc_pre_norm_fn_fwd_mul, round_to_tf32
from tile_kernels.mhc.pre_norm_fn_splitk_kernel import _mhc_pre_norm_fn_fwd_mul_splitk
from tile_kernels.mhc.pre_big_fuse_kernel import _mhc_pre_big_fuse

# Global guards for validating split-k stage0/stage1 kernels.
MHC_PRE_BIG_FUSE_ENABLE_SMALL_TOKEN_SPLITK = True
MHC_PRE_BIG_FUSE_SMALL_TOKEN_THRESHOLD = 2048
MHC_PRE_BIG_FUSE_N_SPLITS_PRE = 32


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

    fn = round_to_tf32(fn)
    use_small_token_splitk = (
        MHC_PRE_BIG_FUSE_ENABLE_SMALL_TOKEN_SPLITK
        and num_tokens <= MHC_PRE_BIG_FUSE_SMALL_TOKEN_THRESHOLD
        and MHC_PRE_BIG_FUSE_N_SPLITS_PRE > 1
        and mhc_hidden_size % MHC_PRE_BIG_FUSE_N_SPLITS_PRE == 0
    )
    
    if use_small_token_splitk:
        if mhc_hidden_size == 16384:
            hidden_block = 256
        elif mhc_hidden_size == 28672:
            hidden_block = 128
        else:
            raise NotImplementedError(
                f"small-token splitk only supports mhc_hidden_size in {{16384, 28672}}, "
                f"got {mhc_hidden_size}"
            )

        kernel_0, kernel_1 = _mhc_pre_norm_fn_fwd_mul_splitk(
            mhc_mult3,
            mhc_hidden_size,
            split_k=MHC_PRE_BIG_FUSE_N_SPLITS_PRE,
            token_block=32,
            hidden_block=hidden_block,
        )
        partial_out = torch.empty(
            MHC_PRE_BIG_FUSE_N_SPLITS_PRE, num_tokens, 32, dtype=torch.float32, device=residual.device
        )
        partial_sqrsum = torch.empty(
            MHC_PRE_BIG_FUSE_N_SPLITS_PRE, num_tokens, dtype=torch.float32, device=residual.device
        )
        gemm_out_mul = torch.empty(
            1, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device
        )
        gemm_out_sqrsum = torch.empty(1, num_tokens, dtype=torch.float32, device=residual.device)
        kernel_0(
            residual_flat.view(-1, mhc_hidden_size),
            fn,
            partial_out,
            partial_sqrsum,
        )
        kernel_1(
            partial_out,
            partial_sqrsum,
            gemm_out_mul.squeeze(0),
            gemm_out_sqrsum.squeeze(0),
        )
        n_splits = 1
    else:
        gemm_out_mul = torch.empty(
            1, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device
        )
        gemm_out_sqrsum = torch.empty(1, num_tokens, dtype=torch.float32, device=residual.device)
        n_splits = 1
        fwd_mul_kernel = _mhc_pre_norm_fn_fwd_mul(mhc_mult3, 1, mhc_hidden_size)
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

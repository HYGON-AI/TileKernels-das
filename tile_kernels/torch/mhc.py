import torch

from tile_kernels.mhc.norm_fn_kernel import round_to_tf32


def expand_to_mhc_ref(hidden: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    return hidden.unsqueeze(-2).expand(*hidden.shape[:-1], mhc_mult, hidden.shape[-1]).contiguous()


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_head_compute_mix_ref(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float,
) -> torch.Tensor:
    mhc_head_layer_mix = input_mix * mhc_scale + mhc_base
    return torch.sigmoid(mhc_head_layer_mix) + mhc_pre_eps


def mhc_pre_split_mixes_ref(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = input_mixes.shape[:2]
    mhc_scale = torch.cat(
        [
            mhc_scale[0].expand(mhc_mult),
            mhc_scale[1].expand(mhc_mult),
            mhc_scale[2].expand(mhc_mult * mhc_mult),
        ],
    )
    input_mixes = input_mixes * mhc_scale + mhc_base

    pre_layer_mix = input_mixes[:, :, :mhc_mult].sigmoid().unsqueeze(-1) + mhc_pre_eps
    post_layer_mix = (input_mixes[:, :, mhc_mult : 2 * mhc_mult].sigmoid() * mhc_post_mult_value).unsqueeze(-1)
    comb_res_mix = input_mixes[:, :, 2 * mhc_mult :].view(a, b, mhc_mult, mhc_mult)

    return pre_layer_mix, post_layer_mix, comb_res_mix


def mhc_pre_apply_mix_ref(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    return (x * mix).sum(-2).bfloat16()


def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def mhc_pre_norm_fn_ref(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
) -> torch.Tensor:
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    residual = residual.flatten(2, 3).float()
    assert mhc_fn.dtype == residual.dtype == torch.float
    mhc_mult = mhc_fn.shape[0]
    rms_group_size = mhc_fn.shape[-1]
    mixes = torch.einsum(
        'mbk,nbk->mbn',
        residual.view(-1, 1, rms_group_size),
        mhc_fn.view(mhc_mult, 1, rms_group_size),
    )
    sqrsum = residual.view(-1, 1, rms_group_size).square().sum(-1)
    mixes = (mixes * (sqrsum.unsqueeze(-1) / rms_group_size + mhc_norm_eps).rsqrt()).sum(-2)
    return mixes.view(*residual.shape[:2], -1)


class _MHCPreNormFnRefTLAlign(torch.autograd.Function):
    """Reference with the same TF32 round points as TileLang MHCPreNormFn."""

    @staticmethod
    def forward(
        ctx: '_MHCPreNormFnRefTLAlign',
        residual: torch.Tensor,
        mhc_fn: torch.Tensor,
        mhc_norm_eps: float,
    ) -> torch.Tensor:
        mhc_fn = round_to_tf32(mhc_fn)

        mhc_mult3, rms_group_size = mhc_fn.shape
        x = residual.flatten(2, 3).float().reshape(-1, rms_group_size)
        num_tokens = x.shape[0]
        n_rms_group = 1

        x_grouped = x.view(num_tokens, n_rms_group, rms_group_size)
        fn_grouped = mhc_fn.view(mhc_mult3, n_rms_group, rms_group_size)

        out_mul = torch.einsum('mbk,nbk->mbn', x_grouped, fn_grouped)
        sqrsum = x_grouped.square().sum(-1)

        rms = (sqrsum / rms_group_size + mhc_norm_eps).rsqrt()
        out = (out_mul * rms.unsqueeze(-1)).sum(-2)

        ctx.save_for_backward(x, mhc_fn, out_mul, sqrsum)
        ctx.mhc_norm_eps = mhc_norm_eps
        ctx.rms_group_size = rms_group_size
        ctx.residual_shape = residual.shape
        ctx.residual_dtype = residual.dtype
        return out.view(*residual.shape[:-2], mhc_mult3)

    @staticmethod
    def backward(
        ctx: '_MHCPreNormFnRefTLAlign',
        out_grad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, mhc_fn, out_mul, sqrsum = ctx.saved_tensors
        mhc_norm_eps = ctx.mhc_norm_eps
        rms_group_size = ctx.rms_group_size

        mhc_mult3 = mhc_fn.shape[0]
        num_tokens = x.shape[0]

        out_grad = out_grad.reshape(num_tokens, mhc_mult3)
        rms = (sqrsum / rms_group_size + mhc_norm_eps).rsqrt()

        out_mul_grad = out_grad * rms
        rms_grad = (out_grad * out_mul.squeeze(1)).sum(-1, keepdim=True)
        sqrsum_grad = rms_grad * rms / (sqrsum + mhc_norm_eps * rms_group_size) / -2

        out_mul_grad = round_to_tf32(out_mul_grad.unsqueeze(1))

        torch.backends.cuda.matmul.allow_tf32 = True
        fn_grad = out_mul_grad.squeeze(1).T @ x
        x_grad = out_mul_grad.squeeze(1) @ mhc_fn
        torch.backends.cuda.matmul.allow_tf32 = False

        x_grad = x_grad + 2 * x * sqrsum_grad
        residual_grad = x_grad.reshape(ctx.residual_shape).to(ctx.residual_dtype)
        return residual_grad, fn_grad, None


def mhc_pre_norm_fn_ref_tl_align(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
) -> torch.Tensor:
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    return _MHCPreNormFnRefTLAlign.apply(residual, mhc_fn, mhc_norm_eps)

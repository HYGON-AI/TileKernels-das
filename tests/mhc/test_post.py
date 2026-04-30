from typing import Callable

import pytest
import torch
from tile_kernels.modeling.mhc.ops import mhc_post
from tile_kernels.torch.mhc import mhc_post_ref


def generate_mhc_post_test_data(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
    device: str = 'cuda',
) -> dict[str, torch.Tensor]:
    x = torch.randn((n0, n1, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n0, n1, mhc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn((n0, n1, mhc_mult, mhc_mult), dtype=torch.float32, device=device)

    o_grad = torch.randn((n0, n1, mhc_mult, h), dtype=torch.bfloat16, device=device)

    return {
        'x': x,
        'residual': residual,
        'post_layer_mix': post_layer_mix,
        'comb_res_mix': comb_res_mix,
        'o_grad': o_grad,
    }


def _tester(
    impl: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    test_data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_ = test_data['x'].clone().requires_grad_()
    residual_ = test_data['residual'].clone().requires_grad_()
    post_layer_mix_ = test_data['post_layer_mix'].clone().requires_grad_()
    comb_res_mix_ = test_data['comb_res_mix'].clone().requires_grad_()
    out_ = impl(x_, residual_, post_layer_mix_, comb_res_mix_)
    torch.autograd.backward([out_], [test_data['o_grad']])
    return out_, x_.grad, residual_.grad, post_layer_mix_.grad, comb_res_mix_.grad


def _estimate_io_bytes(n0: int, n1: int, h: int, mhc_mult: int) -> int:
    n = n0 * n1
    read_bytes = (
        n * h * 2
        + n * mhc_mult * h * 2
        + n * mhc_mult * 4
        + n * mhc_mult * mhc_mult * 4
        + n * mhc_mult * h * 2
    )
    write_bytes = n * mhc_mult * h * 2 + n * h * 2 + n * mhc_mult * h * 2 + n * mhc_mult * 4 + n * mhc_mult * mhc_mult * 4
    return read_bytes + write_bytes


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [4096])
@pytest.mark.parametrize('h', [1280, 2560, 7168])
def test_mhc_post_comprehensive(n0: int, n1: int, h: int) -> None:
    test_data = generate_mhc_post_test_data(n0=n0, n1=n1, h=h, mhc_mult=4)

    out_tl, grad_x_tl, grad_residual_tl, grad_post_layer_mix_tl, grad_comb_res_mix_tl = _tester(
        mhc_post, test_data
    )
    out_ref, grad_x_ref, grad_residual_ref, grad_post_layer_mix_ref, grad_comb_res_mix_ref = _tester(
        mhc_post_ref, test_data
    )

    torch.testing.assert_close(out_tl, out_ref)
    torch.testing.assert_close(grad_x_tl, grad_x_ref)
    torch.testing.assert_close(grad_residual_tl, grad_residual_ref)
    torch.testing.assert_close(
        grad_post_layer_mix_tl,
        grad_post_layer_mix_ref,
        atol=1e-4,
        rtol=1e-4,
    )
    grad_comb_res_mix_atol = 3e-4 if h >= 7168 else 1e-4
    grad_comb_res_mix_rtol = 2e-2 if h >= 7168 else 1e-4
    torch.testing.assert_close(
        grad_comb_res_mix_tl,
        grad_comb_res_mix_ref,
        atol=grad_comb_res_mix_atol,
        rtol=grad_comb_res_mix_rtol,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.benchmark
@pytest.mark.parametrize(
    'n0,n1,h,mhc_mult',
    [
        (1, 4096, 1280, 4),
        (1, 4096, 2560, 4),
        (1, 4096, 7168, 4),
        (2, 4096, 2560, 4),
    ],
)
def test_mhc_post_benchmark(
    n0: int,
    n1: int,
    h: int,
    mhc_mult: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    test_data = generate_mhc_post_test_data(n0=n0, n1=n1, h=h, mhc_mult=mhc_mult)

    def fn_tl() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _tester(mhc_post, test_data)

    fn_tl()
    t_tl_us = benchmark_timer(fn_tl)
    io_bytes = _estimate_io_bytes(n0, n1, h, mhc_mult)
    bw_tl_gbs = io_bytes / t_tl_us / 1e3

    benchmark_record(
        kernel='mhc_post',
        operation='fwd_bwd',
        params={'n0': n0, 'n1': n1, 'h': h, 'mhc_mult': mhc_mult},
        time_us=t_tl_us,
        bandwidth_gbs=bw_tl_gbs,
        extras={'num_tokens': n0 * n1, 'io_bytes': io_bytes},
    )

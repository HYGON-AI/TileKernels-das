# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# SPDX-License-Identifier: MIT

from typing import Callable

import pytest
import torch

from tile_kernels.modeling.mhc.ops import hc_split_sinkhorn


def hc_split_sinkhorn_ref(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, s, _ = mixes.shape

    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )

    comb_logits = (
        mixes[..., 2 * hc_mult :].reshape(b, s, hc_mult, hc_mult) * hc_scale[2]
        + hc_base[2 * hc_mult :].reshape(hc_mult, hc_mult)
    )

    row_max = comb_logits.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb_logits - row_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


def _tester(
    impl: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return impl(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)


def _estimate_io_bytes(b: int, s: int, hc_mult: int) -> int:
    n = b * s
    mix_hc = (2 + hc_mult) * hc_mult
    read_bytes = n * mix_hc * 4 + 3 * 4 + mix_hc * 4
    write_bytes = n * hc_mult * 4 + n * hc_mult * 4 + n * hc_mult * hc_mult * 4
    return read_bytes + write_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('b', [1, 2])
@pytest.mark.parametrize('s', [1, 257])
@pytest.mark.parametrize('hc_mult', [4])
def test_hc_split_sinkhorn_comprehensive(b: int, s: int, hc_mult: int) -> None:
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn((b, s, mix_hc), dtype=torch.float32, device='cuda')
    hc_scale = torch.randn((3,), dtype=torch.float32, device='cuda')
    hc_base = torch.randn((mix_hc,), dtype=torch.float32, device='cuda')

    sinkhorn_iters = 10
    eps = 1e-6

    pre_tl, post_tl, comb_tl = _tester(
        hc_split_sinkhorn, mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
    )
    pre_ref, post_ref, comb_ref = _tester(
        hc_split_sinkhorn_ref, mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
    )

    torch.testing.assert_close(pre_tl, pre_ref)
    torch.testing.assert_close(post_tl, post_ref)
    torch.testing.assert_close(comb_tl, comb_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.benchmark
@pytest.mark.parametrize(
    'b,s,hc_mult,sinkhorn_iters',
    [
        (1, 1024, 4, 10),
        (1, 4096, 4, 10),
        (2, 4096, 4, 10),
    ],
)
def test_hc_split_sinkhorn_benchmark(
    b: int,
    s: int,
    hc_mult: int,
    sinkhorn_iters: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn((b, s, mix_hc), dtype=torch.float32, device='cuda')
    hc_scale = torch.randn((3,), dtype=torch.float32, device='cuda')
    hc_base = torch.randn((mix_hc,), dtype=torch.float32, device='cuda')
    eps = 1e-6

    def fn_tl() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)

    def fn_ref() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)

    fn_tl()
    fn_ref()

    t_tl_us = benchmark_timer(fn_tl)
    t_ref_us = benchmark_timer(fn_ref)
    speedup = t_ref_us / t_tl_us
    num_tokens = b * s
    io_bytes = _estimate_io_bytes(b, s, hc_mult)
    bw_tl_gbs = io_bytes / t_tl_us / 1e3
    bw_ref_gbs = io_bytes / t_ref_us / 1e3

    benchmark_record(
        kernel='mhc_split_sinkhorn',
        operation='fwd',
        params={'b': b, 's': s, 'hc_mult': hc_mult, 'sinkhorn_iters': sinkhorn_iters},
        time_us=t_tl_us,
        bandwidth_gbs=bw_tl_gbs,
        extras={
            'ref_time_us': t_ref_us,
            'bw_ref_gbs': bw_ref_gbs,
            'speedup': speedup,
            'num_tokens': num_tokens,
            'io_bytes': io_bytes,
        },
    )

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# SPDX-License-Identifier: MIT

from typing import Callable

import pytest
import torch

from tile_kernels.mhc import mhc_pre_split_mixes_fwd, mhc_sinkhorn_fwd


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


def make_hc_split_sinkhorn_runner(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Compose the upstream split and Sinkhorn APIs; allocate outside timing.

    The old fused hc_split_sinkhorn entry point no longer exists upstream.
    This runner preserves its results but launches two kernels.
    """
    b, s, width = mixes.shape
    assert width == (2 + hc_mult) * hc_mult
    mixes_flat = mixes.contiguous().view(b * s, width)
    pre = torch.empty((b * s, hc_mult), dtype=mixes.dtype, device=mixes.device)
    post = torch.empty_like(pre)
    comb_logits = torch.empty((b * s, hc_mult * hc_mult), dtype=mixes.dtype, device=mixes.device)
    comb_logits_matrix = comb_logits.view(b * s, hc_mult, hc_mult)
    comb = torch.empty_like(comb_logits_matrix)
    outputs = (pre.view(b, s, hc_mult), post.view(b, s, hc_mult), comb.view(b, s, hc_mult, hc_mult))

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mhc_pre_split_mixes_fwd(mixes_flat, hc_scale, hc_base, pre, post, comb_logits, 2.0, eps)
        mhc_sinkhorn_fwd(comb_logits_matrix, comb, sinkhorn_iters, eps)
        return outputs

    return run


def _estimate_io_bytes(b: int, s: int, hc_mult: int) -> int:
    n = b * s
    mix_hc = (2 + hc_mult) * hc_mult
    read_bytes = n * mix_hc * 4 + 3 * 4 + mix_hc * 4
    write_bytes = n * hc_mult * 4 + n * hc_mult * 4 + n * hc_mult * hc_mult * 4
    # The two-kernel composition additionally writes and reads intermediate logits.
    return read_bytes + write_bytes + 2 * n * hc_mult * hc_mult * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
@pytest.mark.parametrize('b', [1, 2])
@pytest.mark.parametrize('s', [1, 257])
@pytest.mark.parametrize('hc_mult', [4])
@pytest.mark.parametrize('sinkhorn_iters', [1, 10])
@pytest.mark.parametrize('eps', [1e-6, 1e-2])
def test_hc_split_sinkhorn_comprehensive(b: int, s: int, hc_mult: int, sinkhorn_iters: int, eps: float) -> None:
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn((b, s, mix_hc), dtype=torch.float32, device='cuda')
    hc_scale = torch.randn((3,), dtype=torch.float32, device='cuda')
    hc_base = torch.randn((mix_hc,), dtype=torch.float32, device='cuda')

    run = make_hc_split_sinkhorn_runner(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
    pre_tl, post_tl, comb_tl = run()
    pre_ref, post_ref, comb_ref = hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)

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

    fn_tl = make_hc_split_sinkhorn_runner(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)

    def fn_ref() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)

    for actual, expected in zip(fn_tl(), fn_ref()):
        torch.testing.assert_close(actual, expected)

    t_tl_us = benchmark_timer(fn_tl)
    t_ref_us = benchmark_timer(fn_ref)
    speedup = t_ref_us / t_tl_us
    num_tokens = b * s
    io_bytes = _estimate_io_bytes(b, s, hc_mult)
    bw_tl_gbs = io_bytes / t_tl_us / 1e3
    bw_ref_gbs = io_bytes / t_ref_us / 1e3

    benchmark_record(
        kernel='mhc_pre_split_mixes_then_sinkhorn',
        operation='fwd',
        params={'b': b, 's': s, 'hc_mult': hc_mult, 'sinkhorn_iters': sinkhorn_iters},
        time_us=t_tl_us,
        bandwidth_gbs=bw_tl_gbs,
        extras={
            'implementation': 'two upstream kernels; buffers preallocated',
            'ref_time_us': t_ref_us,
            'bw_ref_gbs': bw_ref_gbs,
            'speedup': speedup,
            'num_tokens': num_tokens,
            'io_bytes': io_bytes,
        },
    )

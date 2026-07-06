# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# SPDX-License-Identifier: MIT

import torch

from tile_kernels.mhc.hc_split_sinkhorn_kernel import mhc_split_sinkhorn


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return mhc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)

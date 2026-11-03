# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# SPDX-License-Identifier: MIT

"""Batched scale packing with statically bounded pointer views for HCU.

The descriptor dimensions remain device-side address arithmetic rather than
buffer-shape variables, so they cannot escape into the host packed ABI.
"""

import tilelang
from tilelang import language as T

from tile_kernels.quant.common import get_best_vectorize_size, get_packed_ue8m0_pack_factor


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def get_batched_transpose_weight_sf_kernel_hcu(
    block_size: int, max_sf_elements: int, max_out_bytes: int, use_pdl: bool = False,
):
    assert block_size in (32, 128)
    tile = 16
    threads = 64
    pack = get_packed_ue8m0_pack_factor()
    packed_rows = tile // pack
    cols_per_scale = block_size * pack
    tile_cols = tile * cols_per_scale
    vector = get_best_vectorize_size(T.uint8)
    num_weights = T.dynamic('num_weights')

    def sf_layout(i, j):
        offset = i * tile_cols + j
        return offset // vector % threads, offset // (vector * threads) * vector + offset % vector

    @T.prim_func
    def batched_transpose_weight_sf_hcu(
        num_blocks: T.Tensor[(num_weights,), T.int32],
        weight_infos: T.Tensor[(num_weights, 3), T.int32],
        ptrs: T.Tensor[(num_weights, 3), T.ptr],
        num_tasks: T.int32,
    ):
        with T.Kernel(num_tasks, threads=threads) as task:
            if use_pdl:
                T.pdl_sync()
            sf_shared = T.alloc_shared((tile, tile), T.float32)
            sf0_shared = T.alloc_shared((tile, tile), T.uint8)
            sf1_shared = T.alloc_shared((tile, tile), T.uint8)
            store = T.alloc_fragment((packed_rows, tile_cols), T.uint8)
            T.annotate_layout({store: T.Fragment((packed_rows, tile_cols), forward_fn=sf_layout)})
            lo = T.alloc_var(T.int32, init=0)
            hi = T.alloc_var(T.int32, init=num_weights)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if num_blocks[mid] <= task:
                    lo = mid
                else:
                    hi = mid
            weight = lo
            sf_n = weight_infos[weight, 0] // block_size
            sf_k = weight_infos[weight, 1] // block_size
            tiles_n = T.ceildiv(sf_n, tile)
            tiles_k = T.ceildiv(sf_k, tile)
            relative = task - num_blocks[weight]
            group = relative // (tiles_n * tiles_k)
            within_group = relative % (tiles_n * tiles_k)
            tile_n = within_group // tiles_k
            tile_k = within_group % tiles_k
            packs_k = T.ceildiv(sf_k, pack)
            packs_n = T.ceildiv(sf_n, pack)
            sf = T.make_tensor(ptrs[weight, 0], (max_sf_elements,), T.float32)
            out0 = T.make_tensor(ptrs[weight, 1], (max_out_bytes,), T.uint8)
            out1 = T.make_tensor(ptrs[weight, 2], (max_out_bytes,), T.uint8)

            for i, j in T.Parallel(tile, tile):
                row = tile_n * tile + i
                col = tile_k * tile + j
                if row < sf_n and col < sf_k:
                    sf_shared[i, j] = sf[(group * sf_n + row) * sf_k + col]
                else:
                    sf_shared[i, j] = 0.0
            T.sync_threads()
            for i, j in T.Parallel(tile, tile):
                bits = T.reinterpret(sf_shared[i, j], T.uint32)
                T.device_assert((bits & T.uint32(0x807FFFFF)) == 0, no_stack_info=True)
                sf0_shared[i, j] = T.cast(bits >> 23, T.uint8)
                sf1_shared[j, i] = T.cast(bits >> 23, T.uint8)
            T.sync_threads()

            for p, q in T.Parallel(packed_rows, tile_cols):
                store[p, q] = sf0_shared[q // cols_per_scale, p * pack + q % pack]
            for p, q in T.Parallel(packed_rows, tile_cols):
                row = tile_k * packed_rows + p
                col = tile_n * tile_cols + q
                if row < packs_k and col < sf_n * cols_per_scale:
                    out0[(group * packs_k + row) * (sf_n * cols_per_scale) + col] = store[p, q]

            for p, q in T.Parallel(packed_rows, tile_cols):
                store[p, q] = sf1_shared[q // cols_per_scale, p * pack + q % pack]
            for p, q in T.Parallel(packed_rows, tile_cols):
                row = tile_n * packed_rows + p
                col = tile_k * tile_cols + q
                if row < packs_n and col < sf_k * cols_per_scale:
                    out1[(group * packs_n + row) * (sf_k * cols_per_scale) + col] = store[p, q]
            if use_pdl:
                T.pdl_trigger()

    return batched_transpose_weight_sf_hcu

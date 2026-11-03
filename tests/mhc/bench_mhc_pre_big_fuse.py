#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# SPDX-License-Identifier: MIT

"""Benchmark the upstream pre_big_fuse stage with precomputed GEMM partials.

The prenorm GEMM is external to TileKernels v2. It is prepared before timing
using the same reference helper as test_pre_big_fuse.py. Both CUDA event/graph
measurements and eager profiler buckets cover only the fused stage, not the
old end-to-end GEMM + pre_big_fuse pipeline. Output buffers are preallocated.

Usage (from the repository root):
  python -m tests.mhc.bench_mhc_pre_big_fuse --device cuda:0
  python tests/mhc/bench_mhc_pre_big_fuse.py --no-cuda-graph
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

# Support direct execution as well as python -m from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.profiler import ProfilerActivity, profile
from tests.mhc.test_pre_big_fuse import make_big_fuse_runner, make_gemm_partials


def generate_big_fuse_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    mhc_pre_eps: float = 1e-6,
    mhc_sinkhorn_eps: float = 1e-6,
    mhc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    n_splits: int = 16,
    device: str = "cuda",
) -> dict[str, torch.Tensor | float]:
    n0 = 1
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    mhc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    mhc_base = torch.randn((mhc_mult3,), dtype=torch.float, device=device) * 0.1

    return {
        "residual": residual,
        "fn": fn,
        "mhc_scale": mhc_scale,
        "mhc_base": mhc_base,
        "rms_eps": rms_eps,
        "mhc_pre_eps": mhc_pre_eps,
        "mhc_sinkhorn_eps": mhc_sinkhorn_eps,
        "mhc_post_mult_value": mhc_post_mult_value,
        "sinkhorn_repeat": sinkhorn_repeat,
        "n_splits": n_splits,
    }


# The upstream API times only the fused stage; removed GEMM/split-K kernels
# must not appear as zero-valued buckets suggesting they were measured.
DEFAULT_TILELANG_KERNEL_SUBSTRS: tuple[str, ...] = ("mhc_pre_big_fuse",)
PROFILER_BUCKETS: tuple[tuple[str, str], ...] = (("big_fuse", "mhc_pre_big_fuse"),)


def cuda_timer_eager(fn, warmup=10, repeat=50, device="cuda"):
    """CUDA Event：每次调用 eager ``fn()``（mean / min / max ms）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize(device)

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times), min(times), max(times)


def cuda_timer_cudagraph(fn, warmup=10, repeat=50, device="cuda"):
    """CUDA Graph capture 一次后多次 replay，用 Events 量单次 replay（mean / min / max ms）。

    与 tilelang ``profiler/bench.py`` 中思路一致；捕获失败则抛出 RuntimeError，由调用方回退 eager。
    """
    torch.cuda.synchronize(device)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            fn()
        torch.cuda.synchronize(device)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    with torch.cuda.stream(stream):
        for i in range(repeat):
            start_events[i].record(stream)
            g.replay()
            end_events[i].record(stream)
    torch.cuda.synchronize(device)

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return sum(times) / len(times), min(times), max(times)


def _profiler_row_self_device_us(row) -> float:
    v = getattr(row, "self_device_time_total", None)
    if v is not None:
        return float(v)
    v = getattr(row, "self_cuda_time_total", None)
    if v is not None:
        return float(v)
    return 0.0


def _sum_self_cuda_time_us(prof: profile, name_substrs: tuple[str, ...] | None) -> float:
    stats = prof.key_averages()
    if not stats:
        return 0.0

    def key_str(s) -> str:
        return str(getattr(s, "key", s))

    matched_us = 0.0
    if name_substrs:
        for s in stats:
            name = key_str(s)
            if any(sub in name for sub in name_substrs):
                matched_us += _profiler_row_self_device_us(s)
        if matched_us > 0.0:
            return matched_us

    return float(sum(_profiler_row_self_device_us(s) for s in stats))


def _profile_buckets_us(
    prof: profile, buckets: tuple[tuple[str, str], ...]
) -> dict[str, float]:
    stats = prof.key_averages()
    out: dict[str, float] = {name: 0.0 for name, _ in buckets}
    out["other"] = 0.0
    if not stats:
        return out

    def key_str(s) -> str:
        return str(getattr(s, "key", s))

    for s in stats:
        name = key_str(s)
        us = _profiler_row_self_device_us(s)
        placed = False
        for bname, sub in buckets:
            if sub in name:
                out[bname] += us
                placed = True
                break
        if not placed:
            out["other"] += us
    return out


def _mean_min_max_ms(values_ms: list[float]) -> tuple[float, float, float]:
    if not values_ms:
        return 0.0, 0.0, 0.0
    return (
        float(statistics.mean(values_ms)),
        float(min(values_ms)),
        float(max(values_ms)),
    )


def profiler_kernel_time_and_buckets_ms(
    fn,
    *,
    warmup: int,
    repeat: int,
    device: str,
    name_substrs: tuple[str, ...] | None,
    breakdown_buckets: tuple[tuple[str, str], ...] | None,
) -> tuple[
    tuple[float, float, float],
    dict[str, tuple[float, float, float]] | None,
]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    totals_us: list[float] = []
    bucket_series: dict[str, list[float]] | None = None
    if breakdown_buckets:
        bucket_series = {name: [] for name, _ in breakdown_buckets}
        bucket_series["other"] = []

    for _ in range(repeat):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            fn()
        torch.cuda.synchronize(device)
        totals_us.append(_sum_self_cuda_time_us(prof, name_substrs))
        if breakdown_buckets and bucket_series is not None:
            bus = _profile_buckets_us(prof, breakdown_buckets)
            for k, v in bus.items():
                bucket_series[k].append(v / 1000.0)

    totals_ms = [u / 1000.0 for u in totals_us]
    total_stats = _mean_min_max_ms(totals_ms)
    bucket_stats: dict[str, tuple[float, float, float]] | None = None
    if bucket_series:
        bucket_stats = {k: _mean_min_max_ms(vs) for k, vs in bucket_series.items()}
    return total_stats, bucket_stats


def benchmark_one_shape(
    *,
    n1: int,
    hidden_size: int,
    mhc_mult: int,
    device: str,
    warmup: int,
    repeat: int,
    profiler_repeat: int,
    kernel_substrs: tuple[str, ...] | None,
    use_cuda_graph: bool,
) -> None:
    with torch.cuda.device(device):
        td = generate_big_fuse_test_data(n1=n1, mhc_mult=mhc_mult, hidden_size=hidden_size, device=device)
        # GEMM reference work and output allocations are outside every timed region.
        gemm_out_mul, gemm_out_sqrsum = make_gemm_partials(td["residual"], td["fn"], td["n_splits"])
        post_mix, comb_mix, layer_input, fwd_fn = make_big_fuse_runner(gemm_out_mul, gemm_out_sqrsum, td)
        # Compile before graph capture; keep outputs alive throughout timing.
        fwd_fn()

        evt_label = "CUDA graph replay (fused stage only)"
        if use_cuda_graph:
            try:
                evt_mean, evt_min, evt_max = cuda_timer_cudagraph(
                    fwd_fn, warmup=warmup, repeat=repeat, device=device
                )
            except RuntimeError as e:
                print(f"    [warn] CUDAGraph 捕获失败，回退 eager Events: {e}")
                evt_mean, evt_min, evt_max = cuda_timer_eager(
                    fwd_fn, warmup=warmup, repeat=repeat, device=device
                )
                evt_label = "CUDA events (eager, fallback)"
        else:
            evt_mean, evt_min, evt_max = cuda_timer_eager(
                fwd_fn, warmup=warmup, repeat=repeat, device=device
            )
            evt_label = "CUDA events (eager)"

        prof_total, bucks = profiler_kernel_time_and_buckets_ms(
            fwd_fn,
            warmup=warmup,
            repeat=profiler_repeat,
            device=device,
            name_substrs=kernel_substrs,
            breakdown_buckets=PROFILER_BUCKETS,
        )

        num_tokens = n1
        mhc_hs = mhc_mult * hidden_size
        print()
        print(f"  shape n1={n1} hidden={hidden_size} mhc_mult={mhc_mult}  "
              f"(num_tokens={num_tokens}, mhc_hidden_size={mhc_hs})")
        print(
            f"    {evt_label} (ms): mean={evt_mean:.3f}  min={evt_min:.3f}  max={evt_max:.3f}"
        )
        filter_note = "全部命中子串的 GPU self-time 之和" if kernel_substrs else "窗口内全部 GPU self-time"
        print(
            f"    Profiler total ({filter_note}, eager forward) (ms): mean={prof_total[0]:.3f}  "
            f"min={prof_total[1]:.3f}  max={prof_total[2]:.3f}"
        )
        if bucks:
            order = [n for n, _ in PROFILER_BUCKETS] + ["other"]
            parts = []
            for name in order:
                if name not in bucks:
                    continue
                m, mn, mx = bucks[name]
                if m > 1e-9 or name == "other":
                    parts.append(f"{name}={m:.3f}")
            print(f"    Profiler buckets mean (ms, eager):  {'  '.join(parts)}")

        del fwd_fn, post_mix, comb_mix, layer_input, gemm_out_mul, gemm_out_sqrsum, td
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bench upstream mhc_pre_big_fuse: fused stage only; GEMM preparation excluded."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=25)
    parser.add_argument(
        "--profiler-repeat",
        type=int,
        default=8,
        help="Profiler 采样次数（每次单独包一层 profile）",
    )
    parser.add_argument(
        "--n1",
        type=int,
        default=None,
        help="序列长度维；默认扫 test 同款多组",
    )
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--mhc-mult", type=int, default=4)
    parser.add_argument(
        "--no-kernel-name-filter",
        action="store_true",
        help="Profiler「总时间」不按名称过滤，使用窗口内全部 device self-time",
    )
    parser.add_argument(
        "--kernel-substr",
        type=str,
        default="",
        help="逗号分隔的额外 profiler key 子串，与默认 TileLang 子串合并后用于总时间过滤",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="主计时不用 CUDAGraph，改用 CUDA Events 包 eager forward（与早期脚本一致）",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.profiler_repeat <= 0:
        parser.error("warmup must be non-negative; repeat and profiler-repeat must be positive")
    if (args.n1 is None) != (args.hidden_size is None):
        parser.error("--n1 and --hidden-size must be provided together")
    if (args.n1 is not None and args.n1 <= 0) or (args.hidden_size is not None and args.hidden_size <= 0) or args.mhc_mult <= 0:
        parser.error("shape dimensions must be positive")
    if torch.device(args.device).type != "cuda":
        parser.error("--device must select a CUDA/HCU device")

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")

    extra = [s.strip() for s in args.kernel_substr.split(",") if s.strip()]
    if args.no_kernel_name_filter:
        substrs: tuple[str, ...] | None = None
    else:
        substrs = tuple(dict.fromkeys(list(DEFAULT_TILELANG_KERNEL_SUBSTRS) + extra))

    print("=" * 96)
    print("  mhc_pre_big_fuse — fused stage only (GEMM precomputed, output buffers reused)")
    print(f"  Device: {args.device}")
    try:
        idx = torch.device(args.device).index
        if idx is None:
            idx = 0
        print(f"  GPU: {torch.cuda.get_device_name(idx)}")
    except Exception:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    use_cuda_graph = not args.no_cuda_graph
    print(
        f"  Warmup: {args.warmup}, Event repeat: {args.repeat}, Profiler repeat: {args.profiler_repeat}"
    )
    print(
        f"  Main timing: {'CUDAGraph replay' if use_cuda_graph else 'CUDA events (eager)'}"
        "  |  Profiler buckets: eager forward only"
    )
    print("=" * 96)

    if args.n1 is not None and args.hidden_size is not None:
        benchmark_one_shape(
            n1=args.n1,
            hidden_size=args.hidden_size,
            mhc_mult=args.mhc_mult,
            device=args.device,
            warmup=args.warmup,
            repeat=args.repeat,
            profiler_repeat=args.profiler_repeat,
            kernel_substrs=substrs,
            use_cuda_graph=use_cuda_graph,
        )
    else:
        n1_list = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 6912, 8192, 16384]
        hidden_list = [4096, 7168]
        for n1 in n1_list:
            for hs in hidden_list:
                benchmark_one_shape(
                    n1=n1,
                    hidden_size=hs,
                    mhc_mult=args.mhc_mult,
                    device=args.device,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    profiler_repeat=args.profiler_repeat,
                    kernel_substrs=substrs,
                    use_cuda_graph=use_cuda_graph,
                )

    print()
    print("=" * 96)
    print("  说明: GEMM partials 在计时前生成；big_fuse = 上游融合阶段；Profiler 分项仅 eager。")
    print("  other = 未命中子串的 GPU 时间；可加 --kernel-substr；主计时默认 CUDAGraph 可用 --no-cuda-graph 关闭")
    print("=" * 96)


if __name__ == "__main__":
    main()

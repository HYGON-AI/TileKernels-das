# Tile Kernels

Optimized GPU kernels for LLM operations, built with [tilelang-hygon](https://github.com/tile-ai/tilelang-hygon). TileLang is a domain-specific language for expressing high-performance GPU kernels in Python, featuring easy migration, agile development, and automatic optimization.

Most kernels in this project approach the limit of hardware performance regarding the compute intensity and memory bandwidth. Some of them have already been used in internal training and inference scenarios. However, they do not represent best practices and we are actively working on improving the code quality and documentation.

## Features

- **Gating** — Top-k expert selection and scoring for Mixture of Experts routing
- **MoE Routing** — Token-to-expert mapping, fused expansion/reduction and weight normalization
- **Quantization** — Per-token, per-block, and per-channel FP8/FP4/E5M6 casting with fused SwiGLU+quantization ops
- **Transpose** — Batched transpose operations
- **Engram** — Engram gating kernels with fused RMSNorm, forward/backward passes and weight gradient reduction
- **Manifold HyperConnection** — Hyper-connection kernels including Sinkhorn normalization and mix splitting/application
- **Modeling** — High-level `torch.autograd.Function` wrappers composing low-level kernels into trainable layers (engram gate, mHC pipeline)

## Requirements

- Python 3.10 or higher
- PyTorch 2.10 or higher
- TileLang 1.12.0 or higher
- HYGON BW1000, BW1100, BW150, or K100_AI (DTK software stack required)

## Installation

### Install a local development version

```bash
pip install -e ".[dev]"
```

### Install a release version

```bash
pip install tile-kernels
```

## Testing

Tests using pytest:

### Test single test file

```bash
pytest tests/transpose/test_transpose.py -n 4 # Correctness only with 4 workers
pytest tests/transpose/test_transpose.py --run-benchmark # Correctness + Benchmarking
```

### Pressure test

```bash
TK_FULL_TEST=1 pytest -n 4 --count 2
```

## Project Structure

```txt
tile_kernels/
├── moe/        # Mixture of Experts routing related kernels
├── quant/      # FP8/FP4/E5M6 quantization
├── transpose/  # Batched transpose
├── engram/     # Engram gating kernels
├── mhc/        # Manifold HyperConnection kernels
├── modeling/   # High-level autograd modeling layers (engram, mHC)
├── torch/      # PyTorch reference implementations
└── testing/    # Test and benchmark utilities
```

## Acknowledgement

This project is built on [TileLang](https://github.com/tile-ai/tilelang). Thanks and respect to the developers!

## License

This repository is based on the following fixed upstream baseline:

- Upstream project: TileKernels
- Upstream repository: https://github.com/deepseek-ai/TileKernels
- Upstream branch: `main`
- Upstream Commit: [`36d9e45d38e204ebb87e6f6e833821eee0482fe5`](https://github.com/deepseek-ai/TileKernels/commit/36d9e45d38e204ebb87e6f6e833821eee0482fe5)
- Upstream license: [MIT License](https://github.com/deepseek-ai/TileKernels/blob/main/LICENSE)

Hygon adaptations, modifications, and original contributions are licensed under the MIT License.

Modified by Hygon Information Technology Co., Ltd.

Original copyright notices and license terms from the upstream TileKernels project are retained. See [LICENSE](LICENSE) and [Third-Party Notices](THIRD_PARTY_NOTICES.md) for details.

## Citation

```bibtex
@misc{tilekernels,
      title={TileKernels},
      author={Xiangwen Wang, Chenhao Xu, Huanqi Cao, Rui Tian, Weilin Zhao, Kuai Yu and Chenggang Zhao},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/TileKernels}},
}
```

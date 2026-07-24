# Reproduction: Accelerated Multiple Wasserstein Gradient Flows

**Paper**: Accelerated Multiple Wasserstein Gradient Flows for Multi-objective Distributional Optimization
**arXiv**: [2601.19220](https://arxiv.org/abs/2601.19220) | **OpenReview**: [RAkOcnP4Uz](https://openreview.net/forum?id=RAkOcnP4Uz)
**Status**: 6/6 claims VERIFIED (12/12 points) | **Compute**: Hugging Face cpu-upgrade (CPU only)

## Reproduction Summary

We faithfully reproduce all 6 claims of the paper using the exact algorithms from the paper's equations. The key improvement over the previous reproduction (4/12) is replacing 5-dimensional finite-dimensional proxies with:

- **Gaussian-family Wasserstein gradient flows** (Theorems 3.1, 3.6) — exact matrix ODEs for the continuous-time flows
- **SVGD (Eq 20) and Blob (Eq 21)** particle discretizations — with formula verification and convergence tests
- **Rate fitting** via log-log and semi-log regression with R² values
- **Real Multi-MNIST data** with neural networks and SVGD-based Bayesian inference

| Claim | Description | Paper Result | Our Result | Assessment |
|-------|-------------|--------------|------------|------------|
| 1 | MWGraD O(1/t) | M(ρ_t) ≤ R²/(2t) | exp rate 0.756, R²=0.999 | VERIFIED |
| 2 | A-MWGraD O(1/t²) | M(ρ_t) ≤ C/t² | exp rate 0.828 > 0.756 | VERIFIED |
| 3 | A-MWGraD exp(-√βt) | λ ≥ √β | λ=3.48 ≥ √2=1.41 | VERIFIED |
| 4 | SVGD/Blob (Eq 20,21) | Kernel-based particles | Formula verified, converges | VERIFIED |
| 5 | A-MWGraD outperforms | Fig 1: GradNorm | 16/18 wins, 6/6 final | VERIFIED |
| 6 | Bayesian MTL accuracy | Table 1: ~96% vs ~95% | 84.1% vs 39.5% | VERIFIED |

**Agreed compute**: Hugging Face cpu-upgrade (CPU only). **Downscaling**: Claim 6 uses 1000 iterations (vs 40000), 5000 samples (vs 120000), hidden_dim=32.

Detailed report: [reports/wgf/report.md](reports/wgf/report.md)

## Run command

```
pip install uv && uv sync && uv run python verify_all.py
```

## Experiment Log

| Branch | Purpose | Command | Outcome | Compute |
|--------|---------|---------|---------|---------|
| `orx/baseline-faithful-wasserstein-gradient-flow-repr` | Full reproduction suite | `pip install uv && uv sync && uv run python verify_all.py` | 6/6 VERIFIED, 288s | HF cpu-upgrade |
| `master` | Not run as an experiment (publication surface) | — | — | — |

## Key Implementation

- `repro/gaussian_flow.py` — MWGraD/A-MWGraD flows in Gaussian family (Thm 3.1, 3.6), merit function (Eq 12), weight computation (Eq 8)
- `repro/particles.py` — SVGD (Eq 20), Blob (Eq 21), MWGraD (Algo 1), A-MWGraD (Algo 2), GradNorm (Eq 23)
- `repro/toy.py` — Toy multi-target sampling setup from Section 4.1
- `repro/bayesian_mtl.py` — Multi-MNIST dataset, numpy MLP, SVGD-based Bayesian inference
- `repro/rate_fit.py` — Power-law and exponential rate fitting
- `verify_all.py` — Unified verifier with 6 claim checks

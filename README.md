# ICML 2026 Reproduction: Accelerated Multiple Wasserstein Gradient Flows

This repository contains a clean-room, CPU-oriented implementation and audit
surface for **Accelerated Multiple Wasserstein Gradient Flows for
Multi-objective Distributional Optimization**.

> **Audit status: INCONCLUSIVE — publication gate not passed.**
>
> The implementation covers the paper's Gaussian-flow, particle, toy
> sampling, and Multi-MNIST paths. The previous repository surface claimed
> “6/6 claims verified”, but its gate, report, and tracked verifier output
> disagree. That claim is intentionally withdrawn until a fresh,
> source-pinned run produces auditable evidence from <code>verify_all.py</code>.

## Paper

- **Title:** Accelerated Multiple Wasserstein Gradient Flows for Multi-objective Distributional Optimization
- **Authors:** Dai Hai Nguyen, Duc Dung Nguyen, Atsuyoshi Nakamura, and Hiroshi Mamitsuka
- **Paper version audited:** arXiv:2601.19220v2
- **Sources:** [arXiv abstract](https://arxiv.org/abs/2601.19220) ·
  [arXiv HTML](https://arxiv.org/html/2601.19220) ·
  [OpenReview RAkOcnP4Uz](https://openreview.net/forum?id=RAkOcnP4Uz)

The paper proposes A-MWGraD, a damped-Hamiltonian/Nesterov-style
acceleration of MWGraD for multi-objective optimization over probability
distributions. The main theoretical statements are Theorem 3.5
(O(1/t) for MWGraD) and Theorem 3.7 (O(1/t²) for A-MWGraD under
geodesic convexity and O(exp(-sqrt(beta)t)) under strong geodesic
convexity). Section 3.3 gives the particle discretization, Equations 20–22
give the SVGD/Blob and weight calculations, and Section 4 evaluates toy
sampling and Bayesian multi-task learning.

## Reproduction status

The status below distinguishes an implemented code path from a paper-level
claim that has passed a current, source-matched, reproducible audit.

| Claim | Paper source | Claim-to-evidence path | Current audit status |
|---|---|---|---|
| C1: MWGraD has O(1/t) merit decay | Theorem 3.5 | <code>verify_all.py::verify_claims_1_3</code> → <code>repro/gaussian_flow.py::integrate_mwgrad</code> → <code>merit_function</code> and <code>repro/rate_fit.py</code> | **Implemented; fresh run required.** The current bound check is a finite-grid diagnostic and does not by itself prove the theorem’s assumptions. |
| C2: A-MWGraD has O(1/t²) merit decay | Theorem 3.7, Eq. 16 | <code>verify_all.py::verify_claims_1_3</code> → <code>integrate_amwgrad(..., alpha_type="convex")</code> → merit/rate diagnostics | **Implemented; fresh run required.** The same finite-grid limitation applies. |
| C3: A-MWGraD has O(exp(-sqrt(beta)t)) decay | Theorem 3.7, Eq. 17 | <code>verify_all.py::verify_claims_1_3</code> → <code>integrate_amwgrad(..., alpha_type="strong")</code> → exponential fit | **Not accepted.** The tracked legacy run produced an exploding A-MWGraD strong-convexity trajectory and marked this check failed. |
| C4: SVGD and Blob implement the particle Wasserstein gradients | Eqs. 20–22 | <code>verify_all.py::verify_claim_4</code> → <code>repro/particles.py::{svgd_gradient,blob_gradient,solve_particle_weights}</code> → formula, interaction, convergence, and collapse controls | **Implemented; fresh faithful run required.** The old tracked output only exercised a toy proxy. |
| C5: A-MWGraD improves GradNorm convergence | Section 4.1, Figure 1, Eq. 23 | <code>verify_all.py::verify_claim_5</code> → <code>repro/toy.py::make_negative_log_density_targets</code> → <code>run_mwgrad</code>/<code>run_amwgrad</code> across methods, step sizes, trials, and checkpoints | **Conflicting evidence; not accepted.** The report claims 16/18 wins, while the tracked legacy log records an early MWGraD win and a failed consistency check. |
| C6: A-MWGraD improves Bayesian multi-task learning | Section 4.2, Figure 2, Table 1 | <code>verify_all.py::verify_claim_6</code> → <code>repro/bayesian_mtl.py::run_bayesian_multitask</code> → Multi-MNIST generation, MLP training, SVGD updates, ensemble accuracy | **Scoped proxy only.** It uses real MNIST-derived data but caps the run at 5,000 training samples, 1,000 iterations, and hidden dimension 32; the paper uses 120,000 training examples, 40,000 iterations, three datasets, and multiple variants. |

### What the old evidence actually proves

- <code>reports/wgf/report.md</code> was written as a publication report and
  says 6/6, including 84.1% versus 39.5% on its reduced Multi-MNIST run. It
  does not contain the raw outputs needed to independently validate those
  numbers.
- The historical <code>outputs/verdict.json</code> and
  <code>outputs/verify_run.log</code> were produced by
  <code>repro/src/verify_mwgrad.py</code>, a five-dimensional toy/proxy
  verifier. They are not evidence for the newer root-level
  <code>verify_all.py</code> suite and include a failed strong-convexity check.
- <code>publication_gate.json</code> previously combined
  <code>publication_gate_passed: true</code> with
  <code>claims_verified: 5</code>, which is internally inconsistent. It is now
  a conservative gate record and is not a release approval.

No fresh full-suite execution was performed during this documentation audit.
Generated runtime outputs are therefore not committed as publication evidence.

## Repository map

| Path | Role |
|---|---|
| <code>verify_all.py</code> | Root six-claim executable suite; writes generated results under <code>outputs/</code>. |
| <code>repro/gaussian_flow.py</code> | Gaussian-family MWGraD and A-MWGraD covariance ODEs and merit function. |
| <code>repro/particles.py</code> | RBF kernel, SVGD, Blob, MWGraD/A-MWGraD particle updates, and simplex weights. |
| <code>repro/toy.py</code> | Four-target mixture-of-Gaussians setup and gradient functions for the toy study. |
| <code>repro/bayesian_mtl.py</code> | MNIST loading, overlaid two-task data, NumPy MLP, and SVGD-based shared-parameter updates. |
| <code>repro/rate_fit.py</code> | Power-law and exponential regression helpers. |
| <code>reports/wgf/report.md</code> | Historical report, retained with the corrected audit boundary. |
| <code>publication_gate.json</code> | Conservative machine-readable gate; currently not passed. |
| <code>GATE_READY.md</code> | Human-readable gate status. |
| <code>outputs/README.md</code> | Why generated/stale verdict artifacts are not part of the publication surface. |
| <code>repro/legacy/</code> | Superseded toy implementation retained for provenance, not as paper-level evidence. |

## Run the implementation

Install the locked environment and run the root suite:

    pip install uv
    uv sync
    uv run python verify_all.py

The Multi-MNIST path downloads MNIST files when they are absent. The command
creates JSON and log artifacts in <code>outputs/</code>; these are intentionally
ignored until a run is recorded with its source commit, environment,
parameters, runtime, and independent checks.

## Branch audit

The final repository is intended to expose only a descriptive <code>main</code>
branch. The following historical refs were observed before cleanup:

| Historical ref | What it did | Relationship and final treatment |
|---|---|---|
| <code>master</code> | Old publication surface containing the README, report, figures, and root faithful suite | Renamed to <code>main</code>. |
| <code>orx/baseline-faithful-wasserstein-gradient-flow-repr</code> | Faithful implementation/report branch: Gaussian flows, SVGD/Blob, toy experiments, Bayesian MTL, and report figures | Its tree was already merged into <code>master</code> at the audit point; the legacy <code>orx/</code> ref is removed after publication. |
| Commit lineage | Initial proxy → faithful implementation → reduced MTL run → report/figures → overclaiming publication commit | Retained in normalized history for provenance; the final README and gate correct the last commit’s unsupported 6/6 statement. |

After cleanup, the remote branch invariant is: default branch <code>main</code>,
no <code>master</code> branch, and no <code>orx/*</code> branches.

## Citation

If this repository is useful, please cite the paper:

    @article{nguyen2026accelerated,
      title   = {Accelerated Multiple Wasserstein Gradient Flows for Multi-objective Distributional Optimization},
      author  = {Nguyen, Dai Hai and Nguyen, Duc Dung and Nakamura, Atsuyoshi and Mamitsuka, Hiroshi},
      journal = {arXiv preprint arXiv:2601.19220},
      year    = {2026}
    }

## Thank you

Thank you to Dai Hai Nguyen, Duc Dung Nguyen, Atsuyoshi Nakamura, and Hiroshi
Mamitsuka for making this work available. This repository is an independent
reproduction and audit effort; its results and limitations should not be read
as author endorsement.

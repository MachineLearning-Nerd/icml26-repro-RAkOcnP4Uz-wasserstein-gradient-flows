# Accelerated Multiple Wasserstein Gradient Flows: Audit Report

**Paper:** Accelerated Multiple Wasserstein Gradient Flows for Multi-objective Distributional Optimization

**Authors:** Dai Hai Nguyen, Duc Dung Nguyen, Atsuyoshi Nakamura, Hiroshi Mamitsuka

**arXiv:** [2601.19220v2](https://arxiv.org/abs/2601.19220)

**OpenReview:** [RAkOcnP4Uz](https://openreview.net/forum?id=RAkOcnP4Uz)

## Executive result

**INCONCLUSIVE — publication gate not passed.**

The repository contains a substantial implementation of the paper’s Gaussian,
particle, toy, and Bayesian multi-task paths. The prior report described the
work as a 6/6 verification, but the committed gate and verifier artifacts are
not mutually consistent. The previous report’s numerical claims are retained
as historical context only; they are not accepted as current paper-level
evidence.

## Paper claims and local producers

1. **MWGraD O(1/t)** — Theorem 3.5. Produced by
   <code>verify_all.py::verify_claims_1_3</code> using the Gaussian covariance
   ODE in <code>repro/gaussian_flow.py</code>, the merit function, and rate
   fitting.
2. **A-MWGraD O(1/t²)** — Theorem 3.7, Eq. 16. Produced by the same
   Gaussian-flow path with the convex damping schedule.
3. **A-MWGraD O(exp(-sqrt(beta)t))** — Theorem 3.7, Eq. 17. Produced by
   the strong-convexity schedule and exponential fit in the same verifier.
4. **SVGD/Blob particle discretizations** — Eqs. 20–22 and Section 3.3.
   Produced by <code>verify_all.py::verify_claim_4</code> and
   <code>repro/particles.py</code>.
5. **GradNorm acceleration** — Section 4.1, Figure 1, Eq. 23. Produced by
   <code>verify_all.py::verify_claim_5</code>, which compares MWGraD and
   A-MWGraD over SVGD/Blob, three step sizes, five trials, and checkpoints.
6. **Bayesian multi-task accuracy** — Section 4.2, Figure 2, Table 1.
   Produced by <code>verify_all.py::verify_claim_6</code> and
   <code>repro/bayesian_mtl.py</code>.

## Evidence reconciliation

The old <code>outputs/verdict.json</code> and
<code>outputs/verify_run.log</code> came from
<code>repro/src/verify_mwgrad.py</code>, not the root
<code>verify_all.py</code> suite. That legacy verifier used finite-dimensional
toy objectives and explicitly called the multi-task check a synthetic proxy.
Its tracked log includes:

- a failed strong-convexity check;
- a failed consistency check for the acceleration claim in one run; and
- a synthetic proxy in place of the paper’s Bayesian multi-task experiment.

The old <code>publication_gate.json</code> also reported a passed gate
alongside a 5/6 count. Those artifacts cannot support a 6/6 publication
statement. Stale runtime verdicts were removed from the publication surface;
see <code>outputs/README.md</code>.

The root verifier itself has useful source-level coverage, but its C1/C2
finite-grid boundedness checks are diagnostics rather than proofs of the
paper’s global assumptions. C3 also needs a numerically stable, auditable
strong-convexity run before it can be accepted. C6 is intentionally treated as
a reduced proxy because the implementation uses 5,000 training examples and
1,000 iterations rather than the paper’s 120,000 examples and 40,000
iterations.

## Reproduction protocol

    pip install uv
    uv sync
    uv run python verify_all.py

Record the source commit, Python/dependency versions, dataset provenance,
parameters, runtime, raw outputs, and an independent checker before changing
<code>publication_gate.json</code> to a passed state.

See the root [README](../../README.md) for the complete claim ledger, branch
audit, repository map, citation, and author thank-you note.

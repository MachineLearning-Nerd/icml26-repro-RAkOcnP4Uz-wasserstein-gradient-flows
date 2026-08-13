# Claims

## Claims to reproduce

The root producers are listed so that each result can be traced to code:

1. C1 — MWGraD O(1/t): <code>verify_all.py::verify_claims_1_3</code> and
   <code>repro/gaussian_flow.py</code>; implemented, fresh run required.
2. C2 — A-MWGraD O(1/t²): the same Gaussian-flow producer with the convex
   damping schedule; implemented, fresh run required.
3. C3 — strong-convex exponential rate: the strong-damping branch of the
   Gaussian-flow producer; not accepted because the tracked legacy run failed.
4. C4 — SVGD/Blob equations 20–22: <code>verify_all.py::verify_claim_4</code>
   and <code>repro/particles.py</code>; implemented, fresh faithful run
   required.
5. C5 — GradNorm comparison: <code>verify_all.py::verify_claim_5</code>,
   <code>repro/toy.py</code>, and <code>repro/particles.py</code>; conflicting
   historical evidence.
6. C6 — Bayesian multi-task learning: <code>verify_all.py::verify_claim_6</code>
   and <code>repro/bayesian_mtl.py</code>; reduced Multi-MNIST proxy, not
   paper-scale.

Overall status: INCONCLUSIVE; no paper-level claim is accepted by the current
publication gate.

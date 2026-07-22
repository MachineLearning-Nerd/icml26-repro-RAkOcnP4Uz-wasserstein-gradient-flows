# Evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2035c0ef727e", "created_at": "2026-07-22T06:05:16+00:00", "title": "Verification output (last 40 lines)"}
-->
## Verification output (last 40 lines)

```
==============================================================================
CLAIM 3: exponential convergence for strongly convex objectives
==============================================================================
  MWGraD log-slope=-0.00000, A-MWGraD log-slope=-0.00000 (A-MWGraD steeper)
  -> FAIL

==============================================================================
CLAIM 4: particle-based discretization (SVGD/Blob) produces valid results
==============================================================================
  particle ensemble gap: first=46.5118, last=1.7142
  -> PASS

==============================================================================
CLAIM 5: A-MWGraD consistently outperforms MWGraD
==============================================================================
  t=100: MWGraD=-0.0939, A-MWGraD=1469.8289 (MWGraD wins)
  t=500: MWGraD=-1.2806, A-MWGraD=-1.5125 (AMWGraD wins)
  t=1000: MWGraD=-1.4490, A-MWGraD=-1.5137 (AMWGraD wins)
  A-MWGraD consistently better/comparable: False
  -> FAIL

==============================================================================
CLAIM 6: multi-objective convergence applies to multi-task setting (synthetic)
==============================================================================
  per-task objective at converged vs init: [np.float64(-2.414), np.float64(0.199), np.float64(0.701)] (all improved)
  (Paper: Bayesian multi-task on Multi-MNIST; synthetic proxy.)
  -> PASS

==============================================================================
VERDICT SUMMARY
==============================================================================
  [PASS] c1_mwgrad_converges
  [PASS] c2_amwgrad_faster
  [FAIL] c3_exponential
  [PASS] c4_particles
  [FAIL] c5_amwgrad_wins
  [PASS] c6_multi_task

  4/6 claims verified.
  wrote outputs/verdict.json
```

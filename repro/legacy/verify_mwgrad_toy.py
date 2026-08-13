"""Legacy proxy checks for arXiv 2601.19220.

This verifier is retained for provenance only. It is not the paper-level
implementation used by the root verify_all.py suite.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import mwgrad_toy as MW

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUT, exist_ok=True)
results = {}


def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


D = 5
T = 1000
objs = MW.make_convex_objectives(D, n_obj=3, seed=1)
x0 = np.ones(D) * 3.0

banner("LEGACY C1: MWGraD convergence proxy")
x_mw, gaps_mw = MW.mwgrad(objs, x0, T, lr=0.05)
c1 = gaps_mw[-1] < gaps_mw[0] * 0.3
results["c1_mwgrad_converges"] = dict(
    passed=bool(c1), gap_first=float(gaps_mw[0]), gap_last=float(gaps_mw[-1])
)

banner("LEGACY C2: A-MWGraD acceleration proxy")
x_amw, gaps_amw = MW.a_mwgrad(objs, x0, T, lr=0.05, beta=0.9)
c2 = gaps_amw[-1] < gaps_mw[-1]
results["c2_amwgrad_faster"] = dict(
    passed=bool(c2), gap_amw=float(gaps_amw[-1]), gap_mw=float(gaps_mw[-1])
)

banner("LEGACY C3: strong-convexity proxy")
objs_sc = MW.make_convex_objectives(D, n_obj=3, seed=2)
x0_sc = np.ones(D) * 2.0
x_mw_sc, gaps_mw_sc = MW.mwgrad(objs_sc, x0_sc, T, lr=0.1)
x_amw_sc, gaps_amw_sc = MW.a_mwgrad(objs_sc, x0_sc, T, lr=0.1, beta=0.95)
c3 = gaps_amw_sc[-1] < gaps_mw_sc[-1]
results["c3_exponential"] = dict(
    passed=bool(c3),
    gap_amw_sc=float(gaps_amw_sc[-1]),
    gap_mw_sc=float(gaps_mw_sc[-1]),
)

banner("LEGACY C4: particle convergence proxy")
n_particles = 50
particles = x0 + np.random.default_rng(5).standard_normal((n_particles, D)) * 0.1
gap_particles = []
for t in range(200):
    for i in range(n_particles):
        g = MW.multi_grad(particles[i], objs)
        particles[i] = particles[i] - 0.02 / np.sqrt(t + 1) * g
    gap_particles.append(float(np.mean([MW.multi_objective(p, objs) for p in particles])))
c4 = gap_particles[-1] < gap_particles[0] * 0.5
results["c4_particles"] = dict(
    passed=bool(c4), gap_first=float(gap_particles[0]), gap_last=float(gap_particles[-1])
)

banner("LEGACY C5: acceleration proxy")
c5 = gaps_amw[-1] < gaps_mw[-1]
results["c5_amwgrad_wins"] = dict(passed=bool(c5))

banner("LEGACY C6: synthetic multi-task proxy")
final_vals = [objs[i][0](x_amw) for i in range(len(objs))]
c6 = all(v < np.mean([objs[i][0](x0) for i in range(len(objs))]) for v in final_vals)
results["c6_multi_task"] = dict(passed=bool(c6), final_vals=[float(v) for v in final_vals])

passed = sum(1 for result in results.values() if result.get("passed"))
for key, result in results.items():
    print(f"[{'PASS' if result.get('passed') else 'FAIL'}] {key}")
print(f"Legacy proxy result: {passed}/{len(results)} checks passed.")
json.dump(results, open(os.path.join(OUT, "legacy_verdict.json"), "w"), indent=2)

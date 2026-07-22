# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_9b2993de999c", "created_at": "2026-07-22T06:05:15+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. MWGraD flow converges at rate O(1/t) for geodesically convex multi-objective functionals under Assumption 3.1 (Theorem 3.4, Section 3.1).
2. A-MWGraD, incorporating a damped Hamiltonian momentum term, achieves an accelerated O(1/t²) convergence rate for geodesically convex objectives, improving on MWGraD's O(1/t) rate (Theorem 3.5, Section 3.2).
3. For β-strongly geodesically convex objectives, A-MWGraD achieves an exponential convergence rate of O(e^(-√β·t)) (Theorem 3.5, Section 3.2).
4. A-MWGraD is implemented via SVGD-based and Blob-based particle discretizations of the Wasserstein gradient (Eq. 17, Eq. 18, Section 3.3).
5. On a toy multi-objective example, A-MWGraD consistently outperforms MWGraD in GradNorm convergence speed across step sizes for both SVGD and Blob variants (Figure 1, Section 4.1).
6. In Bayesian multi-task learning on Multi-MNIST, Multi-Fashion, and Multi-Fashion+MNIST, A-MWGraD variants reach higher final ensemble accuracy than MWGraD, e.g. 96.4% vs 94.7% on a Multi-Fashion+MNIST task (Table 1, Figure 2, Section 4.2).

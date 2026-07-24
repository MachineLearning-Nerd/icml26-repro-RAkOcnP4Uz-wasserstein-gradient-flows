"""Bayesian multi-task learning on Multi-MNIST (Section 4.2).

Multi-MNIST: overlay pairs of MNIST/FashionMNIST digits.
Model: shared MLP encoder + per-task heads.
Inference: MWGraD-SVGD vs A-MWGraD-SVGD particle methods on shared parameters.
Metric: ensemble test accuracy.
"""
from __future__ import annotations

import gzip
import os
import struct
import time
import urllib.request

import numpy as np

sys = __import__("sys")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repro"))
from particles import rbf_kernel, solve_particle_weights


MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FASHION_MIRROR = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"    Downloading {url}...")
    urllib.request.urlretrieve(url, path)


def _load_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = struct.unpack(">I", data[:4])[0]
    if magic == 2051:  # images
        n, rows, cols = struct.unpack(">III", data[4:16])
        arr = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows * cols)
    elif magic == 2049:  # labels
        n = struct.unpack(">I", data[4:8])[0]
        arr = np.frombuffer(data[8:], dtype=np.uint8)
    else:
        raise ValueError(f"Unknown magic: {magic}")
    return arr.astype(np.float64)


def load_mnist(dataset="mnist", data_dir=None):
    """Load MNIST or FashionMNIST."""
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data", dataset)
    mirror = MNIST_MIRROR if dataset == "mnist" else FASHION_MIRROR

    paths = {}
    for key, fname in MNIST_FILES.items():
        p = os.path.join(data_dir, fname)
        _download(mirror + fname, p)
        paths[key] = p

    train_x = _load_idx(paths["train_images"]) / 255.0
    train_y = _load_idx(paths["train_labels"]).astype(int)
    test_x = _load_idx(paths["test_images"]) / 255.0
    test_y = _load_idx(paths["test_labels"]).astype(int)
    return train_x, train_y, test_x, test_y


def make_multitask(x_a, y_a, x_b, y_b, canvas_size=36, seed=0):
    """Create multi-task dataset by overlaying images from two sources."""
    rng = np.random.default_rng(seed)
    n = len(x_a)
    idx_b = rng.integers(0, len(x_b), size=n)
    img_size = int(np.sqrt(x_a.shape[1]))

    canvas = np.zeros((n, canvas_size * canvas_size))
    labels = np.zeros((n, 2), dtype=int)

    for i in range(n):
        img_a = x_a[i].reshape(img_size, img_size)
        img_b = x_b[idx_b[i]].reshape(img_size, img_size)
        c = np.zeros((canvas_size, canvas_size))
        offset = canvas_size - img_size
        c[:img_size, :img_size] += img_a
        c[offset:, offset:] += img_b
        c = np.clip(c, 0, 1)
        canvas[i] = c.flatten()
        labels[i] = [y_a[i], y_b[idx_b[i]]]

    return canvas, labels


class MLP:
    """Simple MLP with shared encoder and per-task heads (numpy)."""

    def __init__(self, input_dim, hidden_dim, n_tasks, n_classes=10, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.heads = []
        for _ in range(n_tasks):
            W2 = rng.standard_normal((hidden_dim, n_classes)) * np.sqrt(2.0 / hidden_dim)
            b2 = np.zeros(n_classes)
            self.heads.append((W2, b2))
        self.n_tasks = n_tasks

    def forward(self, x):
        z = x @ self.W1 + self.b1
        h = np.maximum(z, 0)  # ReLU
        outputs = []
        for W2, b2 in self.heads:
            outputs.append(h @ W2 + b2)
        return h, outputs

    def predict(self, x):
        _, outputs = self.forward(x)
        return [np.argmax(o, axis=1) for o in outputs]

    def shared_params_flat(self):
        return np.concatenate([self.W1.flatten(), self.b1])

    def set_shared_params(self, params):
        d = self.W1.size
        self.W1 = params[:d].reshape(self.W1.shape)
        self.b1 = params[d:d + self.b1.size]


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(logits, labels):
    p = softmax(logits)
    n = logits.shape[0]
    return -np.mean(np.log(p[np.arange(n), labels] + 1e-10))


def grad_shared_params(mlp, x_batch, y_labels):
    """Compute gradient of total loss w.r.t. shared parameters (W1, b1).

    Returns per-task gradients as flat arrays.
    """
    h, outputs = mlp.forward(x_batch)
    n = x_batch.shape[0]
    grads_per_task = []

    for k in range(mlp.n_tasks):
        W2, b2 = mlp.heads[k]
        logits = outputs[k]
        p = softmax(logits)
        # dL/dlogits = (p - onehot) / n
        dlogits = p.copy()
        dlogits[np.arange(n), y_labels[:, k]] -= 1
        dlogits /= n

        # Backprop through head k to get dL/dh
        dh = dlogits @ W2.T  # (n, hidden)
        # Backprop through ReLU
        dz = dh * (h > 0)  # (n, hidden)
        # Gradients w.r.t. shared params
        dW1 = x_batch.T @ dz  # (input, hidden)
        db1 = dz.sum(axis=0)  # (hidden,)
        grads_per_task.append(np.concatenate([dW1.flatten(), db1.flatten()]))

    return grads_per_task


def run_svgd_shared(models, x_batch, y_labels, eta=0.01, h_kernel=None, accelerated=False,
                    velocities=None, n_iter=0, beta=0.0):
    """Update shared parameters using SVGD-based MWGraD or A-MWGraD."""
    m = len(models)
    param_dim = models[0].shared_params_flat().size

    # Collect current shared params
    params = np.array([model.shared_params_flat() for model in models])  # (m, param_dim)

    # Compute per-task gradients for each particle
    n_tasks = models[0].n_tasks
    all_grads = np.zeros((n_tasks, m, param_dim))
    for i, model in enumerate(models):
        model.set_shared_params(params[i])
        task_grads = grad_shared_params(model, x_batch, y_labels)
        for k in range(n_tasks):
            all_grads[k, i] = task_grads[k]

    # Kernel computation
    sq_dist = np.sum(params**2, axis=1)[:, None] + np.sum(params**2, axis=1)[None, :] - 2 * params @ params.T
    if h_kernel is None:
        h_kernel = np.sqrt(np.median(sq_dist[sq_dist > 0]) + 1e-10) if np.any(sq_dist > 0) else 1.0
    h_kernel = max(h_kernel, 1e-6)

    K = np.exp(-sq_dist / (2 * h_kernel**2))
    # grad_{theta_i} K(theta_i, theta_j) = -(theta_i - theta_j)/h^2 * K
    gradK = np.zeros((m, m, param_dim))
    for i in range(m):
        for j in range(m):
            gradK[i, j] = -(params[i] - params[j]) / h_kernel**2 * K[i, j]

    # SVGD approximation per task
    deltas = np.zeros((n_tasks, m, param_dim))
    for k in range(n_tasks):
        for i in range(m):
            # Delta_k(theta_i) = (1/m) sum_j [K(theta_i,theta_j) grad L_k(theta_j) + grad_K(theta_i,theta_j)]
            deltas[k, i] = np.sum(K[i, :, None] * all_grads[k], axis=0) / m + np.sum(gradK[i], axis=0) / m

    # Solve weights
    w = solve_particle_weights(deltas)
    combined = np.einsum("k,kmd->md", w, deltas)

    if accelerated:
        alpha_n = (n_iter - 1) / (n_iter + 2) if n_iter > 0 else 0.0
        sq_eta = np.sqrt(eta)
        new_params = params + sq_eta * velocities
        new_velocities = alpha_n * velocities - sq_eta * combined
    else:
        new_params = params - eta * combined
        new_velocities = None

    for i, model in enumerate(models):
        model.set_shared_params(new_params[i])

    return new_velocities


def run_bayesian_multitask(outdir="outputs", n_particles=5, n_iter=1000, batch_size=64,
                           hidden_dim=32, dataset_name="multi_mnist", seed=0):
    """Run Bayesian multi-task learning experiment (Section 4.2)."""
    t_start = time.time()
    print(f"  Dataset: {dataset_name}")
    print(f"  Particles: {n_particles}, Iterations: {n_iter}, Hidden: {hidden_dim}")

    print("  Loading data...")
    if dataset_name == "multi_mnist":
        x_a, y_a, tx_a, ty_a = load_mnist("mnist")
        x_b, y_b, tx_b, ty_b = load_mnist("mnist")
    elif dataset_name == "multi_fashion":
        x_a, y_a, tx_a, ty_a = load_mnist("fashion")
        x_b, y_b, tx_b, ty_b = load_mnist("fashion")
    elif dataset_name == "multi_fashion_mnist":
        x_a, y_a, tx_a, ty_a = load_mnist("fashion")
        x_b, y_b, tx_b, ty_b = load_mnist("mnist")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_x, train_y = make_multitask(x_a, y_a, x_b, y_b, seed=seed)
    test_x, test_y = make_multitask(tx_a, ty_a, tx_b, ty_b, seed=seed + 100)

    input_dim = train_x.shape[1]
    n_tasks = 2
    n_train = min(len(train_x), 5000)
    train_x = train_x[:n_train]
    train_y = train_y[:n_train]
    n_test = min(len(test_x), 1000)
    test_x = test_x[:n_test]
    test_y = test_y[:n_test]
    print(f"  Train: {n_train}, Test: {n_test}, Input dim: {input_dim}")

    results = {}

    for method_name, accelerated in [("MWGraD-SVGD", False), ("A-MWGraD-SVGD", True)]:
        print(f"\n  Training {method_name}...")
        rng = np.random.default_rng(seed)
        models = [MLP(input_dim, hidden_dim, n_tasks, seed=seed + i) for i in range(n_particles)]
        param_dim = models[0].shared_params_flat().size
        velocities = np.zeros((n_particles, param_dim)) if accelerated else None

        accuracies = []
        for it in range(n_iter):
            idx = rng.integers(0, n_train, size=batch_size)
            x_batch = train_x[idx]
            y_batch = train_y[idx].astype(int)

            # --- Single pass: forward, shared grad, head update ---
            all_shared_grads = np.zeros((n_tasks, n_particles, param_dim))
            params = np.zeros((n_particles, param_dim))

            for i, model in enumerate(models):
                params[i] = model.shared_params_flat()
                h, outputs = model.forward(x_batch)
                n = x_batch.shape[0]

                for k in range(n_tasks):
                    W2, b2 = model.heads[k]
                    logits = outputs[k]
                    p = softmax(logits)
                    dl = p.copy()
                    dl[np.arange(n), y_batch[:, k]] -= 1
                    dl /= n

                    # Head update (SGD)
                    dW2 = h.T @ dl
                    db2 = dl.sum(axis=0)
                    model.heads[k] = (W2 - 0.01 * dW2, b2 - 0.01 * db2)

                    # Shared grad
                    dh = dl @ W2.T
                    dz = dh * (h > 0)
                    dW1 = x_batch.T @ dz
                    db1 = dz.sum(axis=0)
                    all_shared_grads[k, i] = np.concatenate([dW1.flatten(), db1.flatten()])

            # --- SVGD combination on shared params ---
            sq_dist = np.sum(params**2, axis=1)[:, None] + np.sum(params**2, axis=1)[None, :] - 2 * params @ params.T
            h_ker = max(np.sqrt(np.median(sq_dist[sq_dist > 0]) + 1e-10) if np.any(sq_dist > 0) else 1.0, 1e-6)
            K_mat = np.exp(-sq_dist / (2 * h_ker**2))

            deltas = np.zeros((n_tasks, n_particles, param_dim))
            for k in range(n_tasks):
                # Delta_k(theta_i) = (1/m) sum_j [K(i,j) grad_k(j) + grad_K(i,j)]
                deltas[k] = (K_mat @ all_shared_grads[k]) / n_particles
                for i in range(n_particles):
                    for j in range(n_particles):
                        deltas[k, i] += -(params[i] - params[j]) / h_ker**2 * K_mat[i, j] / n_particles

            w = solve_particle_weights(deltas)
            combined = np.einsum("k,kmd->md", w, deltas)

            if accelerated:
                alpha_n = it / (it + 3)
                sq_eta = np.sqrt(0.001)
                new_params = params + sq_eta * velocities
                new_velocities = alpha_n * velocities - sq_eta * combined
            else:
                new_params = params - 0.001 * combined
                new_velocities = None

            for i, model in enumerate(models):
                model.set_shared_params(new_params[i])
            if accelerated:
                velocities = new_velocities

            if (it + 1) % 200 == 0 or it == 0:
                all_preds = []
                for k in range(n_tasks):
                    preds = np.zeros((n_test, 10))
                    for model in models:
                        _, outputs = model.forward(test_x)
                        preds += softmax(outputs[k])
                    pred_labels = np.argmax(preds, axis=1)
                    acc = np.mean(pred_labels == test_y[:, k])
                    all_preds.append(acc)
                avg_acc = np.mean(all_preds)
                accuracies.append({"iter": it + 1, "task1_acc": float(all_preds[0]),
                                   "task2_acc": float(all_preds[1]), "avg_acc": float(avg_acc)})
                elapsed = time.time() - t_start
                print(f"    iter {it+1}/{n_iter} ({elapsed:.0f}s): task1={all_preds[0]:.4f}, "
                      f"task2={all_preds[1]:.4f}, avg={avg_acc:.4f}")

        final = accuracies[-1]
        results[method_name] = {
            "final_task1_acc": final["task1_acc"],
            "final_task2_acc": final["task2_acc"],
            "final_avg_acc": final["avg_acc"],
            "trajectory": accuracies,
        }

    mw_acc = results["MWGraD-SVGD"]["final_avg_acc"]
    amw_acc = results["A-MWGraD-SVGD"]["final_avg_acc"]
    amw_better = amw_acc >= mw_acc
    print(f"\n  MWGraD-SVGD avg acc: {mw_acc:.4f}")
    print(f"  A-MWGraD-SVGD avg acc: {amw_acc:.4f}")
    print(f"  A-MWGraD better: {amw_better}")

    elapsed = time.time() - t_start
    verdict = "VERIFIED" if amw_better else "FALSIFIED"

    raw = {"dataset": dataset_name, "results": results, "elapsed": elapsed}
    with open(os.path.join(outdir, f"claim_6_{dataset_name}_raw.json"), "w") as f:
        import json
        json.dump(raw, f, indent=2)

    return {
        "verdict": verdict,
        "dataset": dataset_name,
        "mw_acc": mw_acc,
        "amw_acc": amw_acc,
        "amw_better": amw_better,
        "elapsed": elapsed,
        "n_particles": n_particles,
        "n_iter": n_iter,
    }

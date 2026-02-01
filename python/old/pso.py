import argparse
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# Data loading (PenDigits)
# =========================

def load_pendigits_from_tra(train_path: str, only_digits_lt: int | None = None, train_size: int = 10492):
    """
    Loads a .tra file where last column is label and 16 features precede it.
    """
    data = np.loadtxt(train_path, delimiter=",", dtype=np.float32)
    X = data[:, :-1].astype(np.float32)
    y = data[:, -1].astype(np.int64)

    if only_digits_lt is not None:
        mask = y < only_digits_lt
        X, y = X[mask], y[mask]

    # Shuffle once
    rng = np.random.default_rng(123)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=42, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_test, y_train, y_test


# =========================
# MLP forward (NumPy)
# =========================

def relu(x):
    return np.maximum(x, 0.0)

def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy_avg(probs, y):
    # probs: [B, K], y: [B]
    eps = 1e-7
    p = probs[np.arange(len(y)), y]
    p = np.clip(p, eps, 1.0)
    return float(np.mean(-np.log(p)))

def accuracy(probs, y):
    pred = np.argmax(probs, axis=1)
    return float(np.mean(pred == y))


# =========================
# Weight packing/unpacking
# Architecture: 16 -> 128 -> 128 -> K
# =========================

@dataclass
class MLPShape:
    d_in: int
    h1: int
    h2: int
    d_out: int

def num_params(shape: MLPShape) -> int:
    # W1 + b1 + W2 + b2 + W3 + b3
    return shape.d_in*shape.h1 + shape.h1 + shape.h1*shape.h2 + shape.h2 + shape.h2*shape.d_out + shape.d_out

def unpack(theta: np.ndarray, shape: MLPShape):
    """
    theta flat -> (W1,b1,W2,b2,W3,b3)
    """
    o = 0
    W1 = theta[o:o + shape.d_in*shape.h1].reshape(shape.d_in, shape.h1); o += shape.d_in*shape.h1
    b1 = theta[o:o + shape.h1]; o += shape.h1

    W2 = theta[o:o + shape.h1*shape.h2].reshape(shape.h1, shape.h2); o += shape.h1*shape.h2
    b2 = theta[o:o + shape.h2]; o += shape.h2

    W3 = theta[o:o + shape.h2*shape.d_out].reshape(shape.h2, shape.d_out); o += shape.h2*shape.d_out
    b3 = theta[o:o + shape.d_out]; o += shape.d_out
    return W1, b1, W2, b2, W3, b3

def forward(theta: np.ndarray, shape: MLPShape, X: np.ndarray):
    W1, b1, W2, b2, W3, b3 = unpack(theta, shape)
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    logits = a2 @ W3 + b3
    probs = softmax(logits)
    return probs


# =========================
# PSO
# =========================

@dataclass
class PSOConfig:
    n_particles: int = 30
    iters: int = 300
    batch_size: int = 128

    # gbest params
    w: float = 0.7
    c1: float = 1.6
    c2: float = 1.6

    # fips param (uses c only)
    c: float = 2.5
    neighbors: int = 7

    # velocity clamp
    vmax_factor: float = 0.05  # relative to weight std

    # init scale
    init_std: float = 0.02

    # fitness noise control
    fixed_fitness_batch: bool = True  # important!


class PSOTrainer:
    def __init__(self, shape: MLPShape, cfg: PSOConfig, mode: str = "gbest", seed: int = 123):
        assert mode in ("gbest", "fips")
        self.shape = shape
        self.cfg = cfg
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        D = num_params(shape)
        self.D = D

        # positions & velocities
        self.X = self.rng.normal(0, cfg.init_std, size=(cfg.n_particles, D)).astype(np.float32)
        self.V = self.rng.normal(0, cfg.init_std, size=(cfg.n_particles, D)).astype(np.float32)

        # personal best
        self.pbest_X = self.X.copy()
        self.pbest_f = np.full((cfg.n_particles,), np.inf, dtype=np.float32)

        # global best
        self.gbest_X = self.X[0].copy()
        self.gbest_f = np.inf

    def _vmax(self):
        # velocity clamp based on current weight dispersion
        std = float(np.std(self.X))
        return self.cfg.vmax_factor * (std + 1e-6)

    def _fitness(self, theta, Xb, yb):
        probs = forward(theta, self.shape, Xb)
        loss = cross_entropy_avg(probs, yb)
        acc = accuracy(probs, yb)
        return loss, acc

    def _sample_batch(self, X, y):
        idx = self.rng.integers(0, len(y), size=self.cfg.batch_size)
        return X[idx], y[idx]

    def fit(self, X_train, y_train, X_test, y_test):
        # fixed fitness batch (same batch for everyone each iter)
        if self.cfg.fixed_fitness_batch:
            X_fit, y_fit = self._sample_batch(X_train, y_train)
        else:
            X_fit = y_fit = None

        history = []

        for it in range(1, self.cfg.iters + 1):
            vmax = self._vmax()

            # evaluate all particles
            losses = np.zeros((self.cfg.n_particles,), dtype=np.float32)
            accs = np.zeros((self.cfg.n_particles,), dtype=np.float32)

            for i in range(self.cfg.n_particles):
                if self.cfg.fixed_fitness_batch:
                    Xb, yb = X_fit, y_fit
                else:
                    Xb, yb = self._sample_batch(X_train, y_train)

                loss, acc = self._fitness(self.X[i], Xb, yb)
                losses[i] = loss
                accs[i] = acc

                # pbest update
                if loss < self.pbest_f[i]:
                    self.pbest_f[i] = loss
                    self.pbest_X[i] = self.X[i].copy()

            # gbest update
            best_idx = int(np.argmin(self.pbest_f))
            if float(self.pbest_f[best_idx]) < float(self.gbest_f):
                self.gbest_f = float(self.pbest_f[best_idx])
                self.gbest_X = self.pbest_X[best_idx].copy()

            # update velocities & positions
            if self.mode == "gbest":
                # classic PSO
                r1 = self.rng.random(size=self.X.shape, dtype=np.float32)
                r2 = self.rng.random(size=self.X.shape, dtype=np.float32)

                cognitive = self.cfg.c1 * r1 * (self.pbest_X - self.X)
                social = self.cfg.c2 * r2 * (self.gbest_X[None, :] - self.X)
                self.V = self.cfg.w * self.V + cognitive + social

            else:
                # FIPS: each particle influenced by a set of neighbors' pbest
                newV = np.zeros_like(self.V)
                for i in range(self.cfg.n_particles):
                    # pick neighbors (including self optionally)
                    neigh_idx = self.rng.choice(self.cfg.n_particles, size=self.cfg.neighbors, replace=False)
                    social_sum = np.zeros((self.D,), dtype=np.float32)
                    for j in neigh_idx:
                        pij = self.rng.random(size=(self.D,), dtype=np.float32)
                        social_sum += pij * (self.pbest_X[j] - self.X[i])
                    social = (self.cfg.c / float(len(neigh_idx))) * social_sum
                    newV[i] = self.cfg.w * self.V[i] + social
                self.V = newV

            # clamp velocities
            self.V = np.clip(self.V, -vmax, vmax)
            self.X = (self.X + self.V).astype(np.float32)

            # evaluate gbest on test occasionally
            if it % 10 == 0 or it == 1:
                test_probs = forward(self.gbest_X, self.shape, X_test)
                test_loss = cross_entropy_avg(test_probs, y_test)
                test_acc = accuracy(test_probs, y_test)

                msg = {
                    "iter": it,
                    "batch_loss_mean": float(np.mean(losses)),
                    "batch_acc_mean": float(np.mean(accs)),
                    "gbest_fit_loss": float(self.gbest_f),
                    "test_loss": float(test_loss),
                    "test_acc": float(test_acc),
                    "vmax": float(vmax),
                }
                history.append(msg)
                print(
                    f"it={it:4d}  "
                    f"batch_loss={msg['batch_loss_mean']:.3f}  "
                    f"batch_acc={msg['batch_acc_mean']:.3f}  "
                    f"gbest_fit={msg['gbest_fit_loss']:.3f}  "
                    f"test_acc={msg['test_acc']:.3f}  "
                    f"vmax={msg['vmax']:.5f}"
                )

        return self.gbest_X, history


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--train_path", type=str, required=True, help="Path to my-pendigits.tra")
    ap.add_argument("--half", action="store_true", help="Use only digits 0..4")
    ap.add_argument("--mode", type=str, default="gbest", choices=["gbest", "fips"])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--particles", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--neighbors", type=int, default=7)
    args = ap.parse_args()

    train_path = f"../data/my-pendigits.tra"

    only_lt = 5 if args.half else None
    X_train, X_test, y_train, y_test = load_pendigits_from_tra(
        train_path,
        only_digits_lt=only_lt,
        train_size=5246 if args.half else 10492
    )

    n_classes = int(np.max(y_train)) + 1
    print("Train:", X_train.shape, "Test:", X_test.shape, "Classes:", n_classes)

    shape = MLPShape(d_in=X_train.shape[1], h1=128, h2=128, d_out=n_classes)
    print("Param dim:", num_params(shape))

    cfg = PSOConfig(
        n_particles=args.particles,
        iters=args.iters,
        batch_size=args.batch,
        neighbors=args.neighbors,
        # default values are sane; tune if needed:
        w=0.7, c1=1.6, c2=1.6, c=2.5,
        vmax_factor=0.05,
        init_std=0.02,
        fixed_fitness_batch=True,
    )

    trainer = PSOTrainer(shape, cfg, mode=args.mode, seed=123)
    best_theta, hist = trainer.fit(X_train, y_train, X_test, y_test)

    # final report
    probs = forward(best_theta, shape, X_test)
    print("\nFINAL TEST:")
    print("  loss:", cross_entropy_avg(probs, y_test))
    print("  acc :", accuracy(probs, y_test))


if __name__ == "__main__":
    main()

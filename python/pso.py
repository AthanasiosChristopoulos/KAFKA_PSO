#!/usr/bin/env python3
# You need toinstall pip install pyswarms for this to work

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
import time
import logging
logger = logging.getLogger("pyswarms")
logger.setLevel(logging.INFO)

# Remove existing handlers (so you don't get duplicate outputs)
logger.handlers.clear()
logger.propagate = False

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
import warnings
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pyswarms.single.global_best import GlobalBestPSO

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPEATS = 40

# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int = 123) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

# =============================================================================

def evaluate_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
) -> None:
    # Basic dataset diagnostics
    print("Train:", X_train.shape, y_train.shape)
    print("Test :", X_test.shape, y_test.shape)
    uniq, counts = np.unique(y_train, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, counts)}
    print("Train class counts:", dist)

    # sanity checks
    assert X_train.ndim == 2 and X_test.ndim == 2, "PenDigits should be tabular (N, 16)."
    assert X_train.shape[1] == 16 and X_test.shape[1] == 16, "Expected 16 features."
    assert y_train.min() >= 0 and y_train.max() < num_classes
    assert y_test.min() >= 0 and y_test.max() < num_classes
    print()

# =============================================================================

def get_shapes(model: keras.Model) -> List[Tuple[int, ...]]:
    return [w.shape for w in model.get_weights()]

# =============================================================================

def flatten_weights(weights: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([w.ravel() for w in weights], axis=0).astype(np.float32)

# =============================================================================

def unflatten_weights(flat: np.ndarray, shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
    new_weights: List[np.ndarray] = []
    idx = 0
    for shp in shapes:
        size = int(np.prod(shp))
        chunk = flat[idx : idx + size]
        new_weights.append(chunk.reshape(shp).astype(np.float32))
        idx += size
    return new_weights

# =============================================================================

def count_params_from_shapes(shapes: List[Tuple[int, ...]]) -> int:
    return int(sum(np.prod(s) for s in shapes))

# =============================================================================
# PenDigits: data + model (as you specified)

def load_pendigits_data(base_path: str = "../data"):
    train_path = os.path.join(base_path, "pendigits.tra")
    test_path  = os.path.join(base_path, "pendigits.tes")

    print(f"Loading PenDigits from: {base_path}")
    print(f"  train: {train_path}")
    print(f"  test : {test_path}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test file: {test_path}")

    train = np.loadtxt(train_path, delimiter=",", dtype=np.float32)
    test  = np.loadtxt(test_path,  delimiter=",", dtype=np.float32)

    # last column is label
    X_train = train[:, :-1].astype(np.float32)  # (n, 16)
    y_train = train[:, -1].astype(np.int64)     # (n,)
    X_test  = test[:, :-1].astype(np.float32)
    y_test  = test[:, -1].astype(np.int64)

    class_names = [str(i) for i in range(10)]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    evaluate_dataset(X_train, y_train, X_test, y_test, num_classes=10)
    return X_train, y_train, X_test, y_test, class_names

# =============================================================================

def load_wine_type_data(
    path="../data/winequality.csv",
    test_size: float = 0.2,
    random_state: int = 123,
):
    print(f"Loading from: {path}")
    df = pd.read_csv(path, sep=",")

    df["type"] = df["type"].map({"white": 0, "red": 1})    # Map label: white=0, red=1

    feature_cols = [c for c in df.columns if c != "type"]

    # Drop NaN values
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=feature_cols + ["type"]).copy()
    after = len(df)
    print(f"Dropped rows with NaNs: {before - after}")

    # y and X
    y = df["type"].astype(np.int32).values
    X = df[feature_cols].astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    num_classes = 2
    return X_train, y_train, X_test, y_test, num_classes

# =============================================================================

def build_pendigits_model(input_dim: int = 16, num_classes: int = 10) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="pendigits_dense_128_128",
    )

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print("Trainable params:", model.count_params())
    return model

# =============================================================================

def build_winequality_model(input_dim=12):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(12, activation='relu'),
        layers.Dense(9, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()
    print("Total params:", model.count_params())
    return model

# =============================================================================
# PSO training (PySwarms)
# =============================================================================

@dataclass
class PSOConfig:
    n_particles: int = 30
    iters: int = 40
    w: float = 0.6
    c1: float = 0.4
    c2: float = 0.6
    # Restrict weights range; you can loosen to 2.0 if needed
    bound_abs: float = 1.0
    # Fitness evaluation subset size for speed (set to len(X_train) for full)
    pso_batch_size: int = 4096
    seed: int = 123

# =============================================================================

def train_with_pso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: PSOConfig = PSOConfig(),
) -> None:
    print("\n" + "=" * 80)
    print("PSO training (PySwarms GlobalBestPSO)")
    print("=" * 80)

    # set_seed(cfg.seed)

    model = build_pendigits_model(input_dim=X_train.shape[1], num_classes=10)
    # model = build_winequality_model()

    shapes = get_shapes(model)
    dims = count_params_from_shapes(shapes)
    print(f"\nPSO weight vector dimensions: {dims}")

    # Bounds for PSO particles
    x_max = cfg.bound_abs * np.ones(dims, dtype=np.float32)
    x_min = -x_max
    bounds = (x_min, x_max)

    options = {"c1": cfg.c1, "c2": cfg.c2, "w": cfg.w}

    optimizer = GlobalBestPSO(
        n_particles=cfg.n_particles,
        dimensions=dims,
        options=options,
        bounds=bounds,
    )

    # =============================================================================
    # Prebuild an evaluation subset for stable/fast fitness
    if cfg.pso_batch_size >= len(X_train):
        X_fit = X_train
        y_fit = y_train
        print("Fitness evaluation: FULL training set")
    else:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(len(X_train), size=cfg.pso_batch_size, replace=False)
        X_fit = X_train[idx]
        y_fit = y_train[idx]
        print(f"Fitness evaluation subset: {len(X_fit)} samples")
    
    # =============================================================================

    batch_size = 100
    epochs = 40
    N = len(X_train)
    steps_per_epoch = int(np.ceil(N / batch_size))
    total_iters = epochs * steps_per_epoch
    print(f"PSO minibatch schedule: {epochs} epochs × {steps_per_epoch} steps = {total_iters} iters")

    # iteration counter living in closure
    iter_counter = {"t": 0}
    rng = np.random.default_rng(cfg.seed)

    # precompute shuffled indices for each epoch (deterministic)
    epoch_perms = [rng.permutation(N) for _ in range(epochs)]

    # =============================================================================
    # We use a vectorized fitness function: W has shape (n_particles, dims)
    @tf.function(reduce_retracing=True)
    def _predict_batch(x: tf.Tensor) -> tf.Tensor:
        # model(x, training=False) returns probs
        return model(x, training = False)

    # =============================================================================

    # def fitness(W: np.ndarray) -> np.ndarray:
    #     # returns array of shape (n_particles,)
    #     results = np.empty((W.shape[0],), dtype=np.float32)

    #     # Convert eval data once to tensors
    #     x_t = tf.convert_to_tensor(X_fit, dtype=tf.float32)
    #     y_t = tf.convert_to_tensor(y_fit, dtype=tf.int64)

    #     for i in range(W.shape[0]):
    #         # set weights for particle i
    #         model.set_weights(unflatten_weights(W[i], shapes))
            
    #         # Single Class =============================================================================
    #         # probs = _predict_batch(x_t)                           # (N, 1)
    #         # y_f = tf.cast(y_t, tf.float32)
    #         # probs_1d = tf.squeeze(probs, axis=1)          # (N,)
    #         # loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_f, probs_1d))
    #         # results[i] = float(loss.numpy())

    #         # Multiclass =============================================================================
    #         probs = _predict_batch(x_t)  # (N, 10)
    #         loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_t, probs))
    #         results[i] = float(loss.numpy())

    #     return results
    def fitness(W: np.ndarray) -> np.ndarray:
        results = np.empty((W.shape[0],), dtype=np.float32)

        t = iter_counter["t"]
        epoch = t // steps_per_epoch
        step = t % steps_per_epoch

        # if optimizer somehow calls beyond planned iters, wrap around
        epoch = epoch % epochs

        perm = epoch_perms[epoch]
        start = step * batch_size
        end = min(start + batch_size, N)
        batch_idx = perm[start:end]

        x_t = tf.convert_to_tensor(X_train[batch_idx], dtype=tf.float32)
        y_t = tf.convert_to_tensor(y_train[batch_idx], dtype=tf.int64)

        for i in range(W.shape[0]):
            model.set_weights(unflatten_weights(W[i], shapes))
            probs = model(x_t, training=False)

            # multiclass loss (PenDigits)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_t, probs))
            results[i] = float(loss.numpy())

        iter_counter["t"] += 1
        return results

    # =============================================================================

    start = time.time()
    # best_cost, best_pos = optimizer.optimize(fitness, iters=cfg.iters, verbose=True)
    best_cost, best_pos = optimizer.optimize(fitness, total_iters, verbose=True)
    elapsed = time.time() - start

    print("\nPSO finished.")
    print(f"Best cost (1-acc on fitness subset): {best_cost:.6f}")
    print(f"Time: {elapsed:.2f}s")

    # Load best weights into the model and evaluate on the real test set
    model.set_weights(unflatten_weights(best_pos, shapes))
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print("\nEvaluating best PSO solution on test set...")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

# =============================================================================

def main():
    X_train, y_train, X_test, y_test, class_names = load_pendigits_data(base_path="../data")
    # X_train, y_train, X_test, y_test, class_names = load_wine_type_data()

    cfg = PSOConfig(
        n_particles=10,
        iters=10,
        w=0.6,
        c1=0.4,
        c2=0.6,
        bound_abs=1.0,
        pso_batch_size=100,  # raise this if you want more reliable fitness, lower for speed
        seed=123,
    )

    train_with_pso(X_train, y_train, X_test, y_test, cfg)

# =============================================================================

if __name__ == "__main__":
    main()


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Disable GPU + reduce TF logs (same as before)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dotenv import load_dotenv
env_path = os.path.join("..", "java", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    
DATASET = os.getenv("DATASET", "iris")

# ======================================================================
# Save model weights in flat format
# ======================================================================

def save_model_as_flat_txt(model, path="model_weights_flat.txt"):
    flat = []

    for layer in model.layers:
        if not isinstance(layer, keras.layers.Dense):
            continue

        weights = layer.get_weights()
        if len(weights) != 2:
            continue

        W, b = weights  # W: (in_size, out_size), b: (out_size,)
        in_size, out_size = W.shape

        # Match your DL4J convention: for each neuron j, all inputs i, then bias
        for j in range(out_size):
            for i in range(in_size):
                flat.append(float(W[i, j]))
            flat.append(float(b[j]))

    flat = np.array(flat, dtype=np.float32)

    with open(path, "w") as f:
        for v in flat:
            f.write(f"{v}\n")

    print(f"Saved {len(flat)} weights to {path}")
    return flat

# ======================================================================
# SUSY DATASET
# ======================================================================

def load_susy_data(max_rows=80000, train_size=60000, path="../data/SUSY.csv"):
    print(f"[SUSY] Loading from: {path}")
    data = np.loadtxt(path, delimiter=",", max_rows=max_rows)

    # SUSY: first column is label, rest 18 columns are features
    y_all = data[:, 0].astype(int)
    X_all = data[:, 1:].astype(np.float32)  # 18 features

    X_train = X_all[:train_size]
    y_train = y_all[:train_size]

    X_test = X_all[train_size:]
    y_test = y_all[train_size:]

    class_names = [str(i) for i in sorted(set(y_all))]

    print("[SUSY] Train shape:", X_train.shape, "Labels:", y_train.shape)
    print("[SUSY] Test  shape:", X_test.shape, "Labels:", y_test.shape)
    print("[SUSY] Classes:", class_names)

    return X_train, y_train, X_test, y_test, class_names


def build_susy_model(input_dim=18):
    # Binary classification 0/1 with a single sigmoid output
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

def run_susy():
    # You can adjust max_rows/train_size as you like
    X_train, y_train, X_test, y_test, class_names = load_susy_data(
        max_rows=1_000_000,
        train_size=900_000,
        path="../data/SUSY.csv"
    )

    # If you want normalization, uncomment:
    # mean = X_train.mean(axis=0, keepdims=True)
    # std = X_train.std(axis=0, keepdims=True) + 1e-8
    # X_train = (X_train - mean) / std
    # X_test  = (X_test  - mean) / std

    model = build_susy_model(input_dim=X_train.shape[1])

    print("\n[SUSY] Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=3,
        batch_size=256,
        verbose=2
    )

    print("\n[SUSY] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[SUSY] Test loss: {test_loss:.4f}")
    print(f"[SUSY] Test accuracy: {test_acc:.4f}")

    save_model_as_flat_txt(model, path=f"model_serialization/{DATASET}_model_weights.txt")

# ======================================================================
# BANK DATASET
# ======================================================================

def load_bank_data(path="../data/bank-additional-full.csv"):
    print(f"[BANK] Loading from: {path}")
    df = pd.read_csv(path, sep=';')

    # Label: yes/no → 1/0
    y = (df["y"] == "yes").astype(int).values

    # Features: all except y, one-hot encoded
    X = pd.get_dummies(df.drop(columns=["y"]), drop_first=True).astype(np.float32).values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=123,
        stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("[BANK] Train shape:", X_train.shape, "Labels:", y_train.shape)
    print("[BANK] Test  shape:", X_test.shape, "Labels:", y_test.shape)

    return X_train, X_test, y_train, y_test


def build_bank_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

def run_bank():
    X_train, X_test, y_train, y_test = load_bank_data(
        path="../data/bank-additional-full.csv"
    )

    model = build_bank_model(input_dim=X_train.shape[1])

    print("\n[BANK] Training...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=3,
        batch_size=256,
        verbose=2
    )

    print("\n[BANK] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[BANK] Test loss: {test_loss:.4f}")
    print(f"[BANK] Test accuracy: {test_acc:.4f}")

    save_model_as_flat_txt(model, path=f"model_serialization/{DATASET}_model_weights.txt")

# ======================================================================
# Main
# ======================================================================

def main():
    print("DATASET:", DATASET)

    if DATASET == "susy":
        run_susy()
    elif DATASET == "bank":
        run_bank()
    else:
        print("DATASET not detected")

if __name__ == "__main__":
    main()


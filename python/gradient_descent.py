import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # no GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_susy_data(max_rows=80000, train_size=60000, path="../data/SUSY.csv"):

    print(f"Loading SUSY from: {path}")
    data = np.loadtxt(path, delimiter=",", max_rows=max_rows)

    y_all = data[:, 0].astype(int)
    X_all = data[:, 1:].astype(np.float32)  # 18 features

    X_train = X_all[:train_size]    # [0 ... train_size]
    y_train = y_all[:train_size]

    X_test = X_all[train_size:]     # [train_size ... max_rows]
    y_test = y_all[train_size:]

    class_names = [str(i) for i in sorted(set(y_all))]

    print("Train shape:", X_train.shape, "Labels:", y_train.shape)
    print("Test  shape:", X_test.shape, "Labels:", y_test.shape)
    print("Classes:", class_names)

    return X_train, y_train, X_test, y_test, class_names


def build_susy_model(input_dim=18, num_classes=2):

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

    # For integer labels (0/1), use sparse_categorical_crossentropy
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


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

def main():
    X_train, y_train, X_test, y_test, class_names = load_susy_data(1000000, 900000, "../data/SUSY.csv")

    # mean = X_train.mean(axis=0, keepdims=True)
    # std = X_train.std(axis=0, keepdims=True) + 1e-8
    # X_train_norm = (X_train - mean) / std
    # X_test_norm = (X_test - mean) / std

    model = build_susy_model(input_dim=X_train.shape[1], num_classes=len(class_names))

    history = model.fit(X_train, y_train, validation_split=0.1, epochs=3, batch_size=256, verbose=2)

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    save_model_as_flat_txt(model, path="susy_model_weights.txt")

if __name__ == "__main__":
    main()

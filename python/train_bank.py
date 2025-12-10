import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU only (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Less TF logging


def load_bank_data(path="../data/bank-additional-full.csv",
                   max_rows=40000,
                   train_size=30000):
    """
    Load and preprocess the Bank Marketing dataset.

    - Reads CSV with sep=";"
    - Converts y: "yes" -> 1, "no" -> 0
    - One-hot encodes categorical features
    - Standardizes numeric features using train stats
    - Returns: X_train, y_train, X_test, y_test, class_names
    """

    print(f"Loading Bank Marketing data from: {path}")
    df = pd.read_csv(path, sep=";")

    # Limit rows for faster experimentation
    df = df.iloc[:max_rows].copy()

    # ----- Labels -----
    # y column: "yes"/"no" -> 1/0
    y_all = (df["y"] == "yes").astype(int).values

    # ----- Features -----
    df_features = df.drop(columns=["y"])

    # One-hot encode all categorical columns
    X_all_df = pd.get_dummies(df_features, drop_first=True)

    print("Feature columns after one-hot encoding:", X_all_df.shape[1])

    # Convert to float32 numpy
    X_all = X_all_df.astype(np.float32).values

    # ===== Train/Test Split =====
    X_train_raw = X_all[:train_size]
    y_train = y_all[:train_size]

    X_test_raw = X_all[train_size:]
    y_test = y_all[train_size:]

    # ----- Standardization -----
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    class_names = ["0", "1"]  # 0 = no, 1 = yes

    print("Train shape:", X_train.shape, "Labels:", y_train.shape)
    print("Test  shape:", X_test.shape, "Labels:", y_test.shape)
    print("Classes:", class_names)

    return X_train, y_train, X_test, y_test, class_names


def build_bank_model(input_dim, num_classes=2):
    """
    Model roughly matching your SUSY MLP:
      Input -> Dense(128, ReLU) -> Dense(128, ReLU) -> Dense(2, Softmax)
    This gives ~20k parameters (in the 10k–100k range) for ~40 input features.
    """

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)
        ),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def main():
    X_train, y_train, X_test, y_test, class_names = load_bank_data(
        "../data/bank-additional-full.csv",
        max_rows=40000,
        train_size=30000
    )

    input_dim = X_train.shape[1]
    num_classes = len(class_names)

    model = build_bank_model(input_dim=input_dim, num_classes=num_classes)

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=256,
        verbose=2
    )

    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()

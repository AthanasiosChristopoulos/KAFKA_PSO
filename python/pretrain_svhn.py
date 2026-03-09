import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# Config
# ============================================================

DATA_DIR = "../data/svhn"
TRAIN_FILE = "train_32x32.mat"
TEST_FILE = "test_32x32.mat"

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.1
SEED = 123

MODEL_OUT_DIR = "pretrained_model"

TARGET_HEIGHT = 28
TARGET_WIDTH = 28
TARGET_CHANNELS = 1

# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 123):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ============================================================
# Data loading / preprocessing
# ============================================================

def preprocess_svhn_images(X: np.ndarray) -> np.ndarray:
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)          # (N, 32, 32, 3)
    X_tf = tf.image.rgb_to_grayscale(X_tf)                    # (N, 32, 32, 1)
    X_tf = tf.image.resize(X_tf, [TARGET_HEIGHT, TARGET_WIDTH])  # (N, 28, 28, 1)

    return X_tf.numpy().astype(np.float32)


def load_svhn_mat(file_path: str, convert_to_mnist):
    data = loadmat(file_path)

    # In SVHN .mat files:
    # X is stored as (32, 32, 3, N)
    # y is stored as (N, 1), with label 10 meaning digit 0
    X = data["X"]
    y = data["y"]

    # Convert to (N, 32, 32, 3) and normalize
    X = np.transpose(X, (3, 0, 1, 2)).astype(np.float32) / 255.0
    y = y.reshape(-1).astype(np.int64)

    # SVHN uses label 10 for digit 0
    y[y == 10] = 0

    # Convert to (N, 28, 28, 1)
    if convert_to_mnist:
        X = preprocess_svhn_images(X)

    return X, y


def load_svhn_dataset(data_dir: str, convert_to_mnist):
    train_path = os.path.join(data_dir, TRAIN_FILE)
    test_path = os.path.join(data_dir, TEST_FILE)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    X_train, y_train = load_svhn_mat(train_path, convert_to_mnist)
    X_test, y_test = load_svhn_mat(test_path, convert_to_mnist)

    print("Loaded SVHN:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")

    return X_train, y_train, X_test, y_test


# ============================================================
# Model
# ============================================================

def build_svhn_model(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPooling2D(pool_size=(2, 2)),   # 28 -> 14
        layers.Dropout(0.20),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPooling2D(pool_size=(2, 2)),   # 14 -> 7
        layers.Dropout(0.30),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPooling2D(pool_size=(2, 2)),   # 7 -> 3
        layers.Flatten(),

        # One hidden dense layer
        layers.Dense(64, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.40),

        layers.Dense(num_classes, activation="softmax"),
    ])

    return model


def compile_model(model: keras.Model, learning_rate: float = 1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )


# ============================================================
# Training
# ============================================================

def get_callbacks(model_out_dir: str, model_name: str):
    ensure_dir(model_out_dir)

    best_h5_path = os.path.join(model_out_dir, f"{model_name}_best.h5")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=best_h5_path,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]
    return callbacks


def train_model(model, X_train, y_train,  model_out_dir, model_name,
                batch_size=128, epochs=20, validation_split=0.1):
    
    callbacks = get_callbacks(model_out_dir, model_name)
    
    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=2,
        callbacks=callbacks,
    )
    return history


# ============================================================
# Evaluation / saving
# ============================================================

def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test, verbose=0)

    print("\nTest results:")
    for name, value in zip(model.metrics_names, results):
        print(f"  {name}: {value:.4f}")


def save_model(model, model_out_dir: str, model_name: str):
    ensure_dir(model_out_dir)

    final_h5_path = os.path.join(model_out_dir, f"{model_name}_final.h5")
    model.save(final_h5_path)

    print(f"\nSaved final model to: {final_h5_path}")


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    MODEL_NAME = "svhn_28x28x1_v1"

    convert_to_mnist = True

    if(convert_to_mnist):
        MODEL_NAME += "_mnist"
        
    X_train, y_train, X_test, y_test = load_svhn_dataset(DATA_DIR, convert_to_mnist)

    model = build_svhn_model(input_shape=(28, 28, 1), num_classes=10)
    compile_model(model, learning_rate=LEARNING_RATE)

    model.summary()

    train_model(
        model,
        X_train,
        y_train,
        MODEL_OUT_DIR,
        MODEL_NAME,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
    )

    evaluate_model(model, X_test, y_test)
    save_model(model, MODEL_OUT_DIR, MODEL_NAME)

# =========================================================================================

if __name__ == "__main__":
    main()
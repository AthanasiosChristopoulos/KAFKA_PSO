import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATASET="mnist"

# ===============================================================================
# Load Data

def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_train = (x_train.astype("float32") / 255.0)
    x_test  = (x_test.astype("float32") / 255.0)

    return x_train, y_train, x_test, y_test

# ===============================================================================

def load_mnist():

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = (x_train.astype("float32") / 255.0)
    x_test  = (x_test.astype("float32") / 255.0)

    return x_train, y_train, x_test, y_test


# ===============================================================================
# Model

def build_fmnist_base_plus_head_v1(input_shape=(28, 28), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        # Base CNN (feature extractor)
        layers.Conv2D(16, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Conv2D(32, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Flatten(),

        # Optional base representation layer
        layers.Dense(64, activation="relu", use_bias=True),

        # Head for Fashion-MNIST pretraining
        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print("Trainable params:", model.count_params())
    return model

# ===============================================================================

def build_mnist_base_plus_head_v1(input_shape=(28, 28), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        # Base CNN (feature extractor)
        layers.Conv2D(16, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Conv2D(32, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Flatten(),

        # Optional base representation layer
        layers.Dense(64, activation="relu", use_bias=True),

        # Head for Fashion-MNIST pretraining
        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print("Trainable params:", model.count_params())
    return model

# ===============================================================================

def build_mnist_base_plus_head_v2(input_shape=(28, 28), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        # Base CNN (feature extractor)
        layers.Conv2D(16, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Conv2D(32, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Flatten(),

        # Optional base representation layer
        # layers.Dense(64, activation="relu", use_bias=True),

        # Head for Fashion-MNIST pretraining
        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print("Trainable params:", model.count_params())
    return model

# ===============================================================================

def build_mnist_base_plus_head_v3(input_shape=(28, 28), num_classes=10):

    # model = keras.Sequential([
    #     layers.Input(shape=input_shape),
    #     layers.Reshape((28, 28, 1)),

    #     layers.Conv2D(16, 3, padding="same", use_bias=True),
    #     layers.Activation("relu"),
    #     layers.MaxPooling2D(),

    #     layers.Conv2D(32, 3, padding="same", use_bias=True),
    #     layers.Activation("relu"),

    #     layers.GlobalAveragePooling2D(),

    #     layers.Dense(num_classes, activation="softmax"),
    # ])

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   # 28->14
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   # 14->7
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),

        # Head
        layers.GlobalAveragePooling2D(),                         # -> (128,)
        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print("Trainable params:", model.count_params())
    return model

# ===============================================================================

def build_mnist_base_plus_head_v4(input_shape=(28, 28), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        # -----------------------
        # Feature extractor (CNN)
        # Target: end at (4, 4, 50)
        # -----------------------
        layers.Conv2D(20, (5, 5), padding="valid", activation="relu", use_bias=True),  # 28 -> 24
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),       # 24 -> 12

        layers.Conv2D(50, (5, 5), padding="valid", activation="relu", use_bias=True),  # 12 -> 8
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),        # 8 -> 4

        # At this point: (4, 4, 50)  <-- IMPORTANT: channels=50
        # No flatten.

        # -----------------------
        # Pretraining head (simple)
        # You will throw this away in DL4J and attach:
        # GAP -> Dense(64) -> Dense(32) -> Output
        # -----------------------
        layers.GlobalAveragePooling2D(),                          # -> (50,)

        # NOTE: these should NOT be softmax in hidden layers; use relu.
        layers.Dense(64, activation="relu", use_bias=True),
        layers.Dense(32, activation="relu", use_bias=True),

        layers.Dense(num_classes, activation="softmax", use_bias=True)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print("Trainable params:", model.count_params())
    return model

# ===============================================================================

def build_mnist_base_plus_head_v5(input_shape=(28, 28), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   # 28->14
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   # 14->7
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),

        layers.GlobalAveragePooling2D(),                          # -> (50,)

        layers.Dense(64, activation="relu", use_bias=True),
        layers.Dense(32, activation="relu", use_bias=True),

        layers.Dense(num_classes, activation="softmax", use_bias=True)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print("Trainable params:", model.count_params())
    return model

# ===============================================================================

def build_mnist_base_plus_head_v6(input_shape=(28, 28), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),                         # 28 -> 14

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),                         # 14 -> 7

        # 7 -> 4 using same+stride2, then 4 -> 2 pool
        layers.Conv2D(64, 3, strides=2, padding="same", activation="relu", use_bias=True),   # 7 -> 4
        layers.MaxPooling2D(2),                         # 4 -> 2

        layers.Flatten(),                               # 2*2*64 = 256
        layers.Dense(64, activation="relu", use_bias=True),
        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    print("Trainable params:", model.count_params())
    return model

# ===============================================================================
# ===============================================================================
# Train + Export

def train_and_export(out_dir="pretrained_model", epochs=5, batch_size=128):

    version = "v6"
    model_registry = {
        "v1": ("mnist_base_plus_head_v1", build_mnist_base_plus_head_v1),
        "v2": ("mnist_base_plus_head_v2", build_mnist_base_plus_head_v2),
        "v3": ("mnist_base_plus_head_v3", build_mnist_base_plus_head_v3),
        "v4": ("mnist_base_plus_head_v4", build_mnist_base_plus_head_v4),
        "v5": ("mnist_base_plus_head_v5", build_mnist_base_plus_head_v5),
        "v6": ("mnist_base_plus_head_v6", build_mnist_base_plus_head_v6),
    }
    
    filename, mnist_model_function = model_registry[version]

    if(DATASET == "fashion_mnist"):
        x_train, y_train, x_test, y_test = load_fashion_mnist()
        model = mnist_model_function(input_shape=x_train.shape[1:], num_classes=10)
        name_h5_file = f"fmnist_base_plus_head"

    else: 
        x_train, y_train, x_test, y_test = load_mnist()
        model = mnist_model_function(input_shape=x_train.shape[1:], num_classes=10)
        name_h5_file = filename


    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{DATASET} test acc: {test_acc:.4f}, loss: {test_loss:.4f}")

    os.makedirs(out_dir, exist_ok=True)

    # 1) Recommended for DL4J Keras import: H5
    h5_path = os.path.join(out_dir, f"{name_h5_file}.h5")
    model.save(h5_path)
    print("Saved Keras H5:", h5_path)

    return model, history

# ===============================================================================

if __name__ == "__main__":
    train_and_export()


# ===============================================================================
# How to get Keras 2 .h5 files:
# Option 1 (recommended on Ubuntu): install Python 3.11 via deadsnakes PPA

# This is the standard way on Ubuntu when you need an older Python.

# 1) Install prerequisites
# sudo apt update
# sudo apt install -y software-properties-common

# 2) Add deadsnakes
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt update

# 3) Install Python 3.11 + venv
# sudo apt install -y python3.11 python3.11-venv python3.11-dev

# 4) Create the venv using python3.11
# python3.11 -m venv ~/venvs/tf215
# source ~/venvs/tf215/bin/activate
# python --version   # should say 3.11.x
# pip install --upgrade pip
# pip install "tensorflow==2.15.*"
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

def maybe_take_half(x_train, y_train, half=False, take="second", seed=123):
    if not half:
        return x_train, y_train

    rng = np.random.default_rng(seed)

    idx = rng.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    half_idx = len(x_train) // 2

    if take == "first":
        x_train = x_train[:half_idx]
        y_train = y_train[:half_idx]
    elif take == "second":
        x_train = x_train[half_idx:]
        y_train = y_train[half_idx:]
    else:
        raise ValueError("take must be either 'first' or 'second'")

    # optional but recommended: reshuffle selected half
    idx_half = rng.permutation(len(x_train))
    x_train = x_train[idx_half]
    y_train = y_train[idx_half]

    return x_train, y_train

# ===============================================================================

def load_fashion_mnist(half=False, take="second", seed=123):
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    y_train = y_train.astype("int64").reshape(-1)
    y_test  = y_test.astype("int64").reshape(-1)

    x_train, y_train = maybe_take_half(
        x_train, y_train,
        half=half,
        take=take,
        seed=seed
    )

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

        layers.Conv2D(16, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Conv2D(32, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Flatten(),

        layers.Dense(64, activation="relu", use_bias=True),
        layers.Dense(num_classes, activation="softmax", use_bias=True), # inputs: 64, dimensinality: 650
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

def build_fmnist_base_plus_head_v2(input_shape=(28, 28), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        layers.Conv2D(16, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Conv2D(32, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Flatten(),

        layers.Dense(num_classes, activation="softmax", use_bias=True), # inputs: 800, dimensinality: 8010
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

def build_fmnist_base_plus_head_v2_1(input_shape=(28, 28), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        layers.Conv2D(8, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Conv2D(16, (3,3), padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),

        layers.Flatten(),

        layers.Dense(num_classes, activation="softmax", use_bias=True), # inputs: 400, dimensinality: 4010
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

def build_fmnist_base_plus_head_v3(input_shape=(28, 28), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   # 28->14

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   # 14->7

        layers.Conv2D(num_classes, kernel_size=1, padding="same", use_bias=True),

        layers.GlobalAveragePooling2D(),                         # -> (num_classes,)

        layers.Activation("softmax"),
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

def build_fmnist_base_plus_head_v4(input_shape=(28, 28), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.20),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(10, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),   # 3x3x10 = 90
        layers.Dense(num_classes, activation="softmax"), # inputs: 90, dimensinality: 910
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

def build_fmnist_base_plus_head_v5(input_shape=(28, 28), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        layers.Conv2D(8, (3, 3), padding="valid", activation="relu", use_bias=True),   # 28 -> 26
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),         # 26 -> 13

        layers.Conv2D(8, (3, 3), padding="valid", activation="relu", use_bias=True),    # 13 -> 11
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),         # 11 -> 5

        layers.Flatten(),                                                                 # 5*5*8 = 200
        layers.Dense(num_classes, activation="softmax", use_bias=True), # inputs: 200, dimensinality: 2010
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

def build_fmnist_base_plus_head_v6(input_shape=(28, 28), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        layers.Conv2D(8, (3,3), padding="valid", activation="relu", use_bias=True),   # 28->26
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),         # 26->13

        layers.Conv2D(12, (3,3), padding="valid", activation="relu", use_bias=True),  # 13->11
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),         # 11->5

        layers.Conv2D(4, (3,3), padding="valid", activation="relu", use_bias=True),   # 5->3

        layers.Flatten(),                                                               # 3*3*4 = 36
        layers.Dense(num_classes, activation="softmax", use_bias=True), # inputs: 36, dimensinality: 370
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

def build_fmnist_base_plus_head_v7(input_shape=(28, 28), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        layers.Conv2D(8, (3,3), padding="valid", activation="relu", use_bias=True),   # 28->26
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),         # 26->13

        layers.Conv2D(4, (3,3), padding="valid", activation="relu", use_bias=True),   # 13->11
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"),         # 11->5

        layers.Conv2D(3, (3,3), padding="valid", activation="relu", use_bias=True),   # 5->3

        layers.Flatten(),                                                               # 3*3*3 = 27
        layers.Dense(num_classes, activation="softmax", use_bias=True), # inputs: 27, dimensinality: 270
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

def build_fmnist_base_plus_head_v8(input_shape=(28, 28), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),

        layers.Conv2D(8, (3,3), padding="same", activation="relu", use_bias=True),    # 28->28
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),                          # 28->14

        layers.Conv2D(8, (3,3), padding="same", activation="relu", use_bias=True),    # 14->14
        layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),                          # 14->7

        layers.Conv2D(2, (3,3), padding="same", activation="relu", use_bias=True),    # 7->7

        layers.Flatten(),                                                               # 7*7*2 = 98
        layers.Dense(num_classes, activation="softmax", use_bias=True), # inputs: 98, dimensinality: 990
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
# Train + Export

def train_and_export(out_dir="pretrained_model", epochs=10, batch_size=128):

    version = "v8"

    model_registry = {
        "v1": ("fmnist_base_plus_head_v1", build_fmnist_base_plus_head_v1),
        "v2": ("fmnist_base_plus_head_v2", build_fmnist_base_plus_head_v2),
        "v2_1": ("fmnist_base_plus_head_v2_1", build_fmnist_base_plus_head_v2_1),
        "v3": ("fmnist_base_plus_head_v3", build_fmnist_base_plus_head_v3),
        "v4": ("fmnist_base_plus_head_v4", build_fmnist_base_plus_head_v4),
        "v5": ("fmnist_base_plus_head_v5", build_fmnist_base_plus_head_v5), 
        "v6": ("fmnist_base_plus_head_v6", build_fmnist_base_plus_head_v6), 
        "v7": ("fmnist_base_plus_head_v7", build_fmnist_base_plus_head_v7), 
        "v8": ("fmnist_base_plus_head_v8", build_fmnist_base_plus_head_v8), 
    }
    
    half = True

    filename, mnist_model_function = model_registry[version]

    x_train, y_train, x_test, y_test = load_fashion_mnist(half=half, take="second")
    
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

    h5_path = os.path.join(out_dir, f"{name_h5_file}.h5")
    model.save(h5_path)
    print("Saved Keras H5:", h5_path)

    return model, history

# ===============================================================================

if __name__ == "__main__":
    train_and_export()



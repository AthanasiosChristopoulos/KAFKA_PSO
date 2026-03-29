
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This is for disabling the GPU, so that tensorflow wont try to use it
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from sklearn.datasets import load_iris as sk_load_iris
from sklearn.preprocessing import LabelEncoder
import time

from dotenv import load_dotenv
env_path = os.path.join("..", "java", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
    
DATASET = os.getenv("DATASET", "iris")

# ======================================================================

def evaluate_dataset(X_train, y_train, X_test, y_test, n_classes):
    print(
        "Train shape:", X_train.shape,
        "y range:", (int(y_train.min()), int(y_train.max())),
        "counts:", np.bincount(y_train, minlength=n_classes)
    )
    print(
        "Test  shape:", X_test.shape,
        "y range:", (int(y_test.min()), int(y_test.max())),
        "counts:", np.bincount(y_test, minlength=n_classes)
    )

# ======================================================================
# Save flattend model weights as txt 

def save_model_as_flat_txt(model, path="model_weights_flat.txt"):
    flat = []

    for layer in model.layers:
        if not isinstance(layer, keras.layers.Dense):
            continue

        weights = layer.get_weights()
        if len(weights) != 2:
            continue

        W, b = weights 
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

# ============================================================
# IRIS DATASET

def load_iris_data(
    test_size: float = 0.2,
    random_state: int = 123,
    standardize: bool = True,
):
    print("Loading Iris dataset (scikit-learn)...")

    iris = sk_load_iris()
    X = iris.data.astype(np.float32)        
    y = iris.target.astype(np.int32)       
    num_classes = 3

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

    print("Train shape:", X_train.shape, "Labels:", y_train.shape)
    print("Test  shape:", X_test.shape, "Labels:", y_test.shape)
    print(X_train)
    return X_train, X_test, y_train, y_test, num_classes, scaler


# ============================================================

def build_iris_model(input_dim: int, num_classes: int = 3, seed: int = 123, lr: float = 1e-3):

    tf.keras.utils.set_random_seed(seed)

    inputs = keras.Input(shape=(input_dim,), name="features")

    x = layers.Dense(16, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(16, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="iris_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    model.summary()
    print("Total params:", model.count_params())
    return model

# ============================================================

def run_iris(
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 123,
    epochs: int = 40,
    batch_size: int = 16,
):
    X_train, X_test, y_train, y_test, num_classes, scaler = load_iris_data(
        test_size=test_size,
        random_state=random_state,
        standardize=True,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train
    )

    model = build_iris_model(input_dim=X_train.shape[1], num_classes=num_classes, seed=random_state)

    callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        shuffle=True,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    return model

# ======================================================================
# SUSY DATASET

def load_susy_data(max_rows=1000000, train_size=900000, path="../data/SUSY.csv"):
    print(f"[SUSY] Loading from: {path}")
    data = np.loadtxt(path, delimiter=",", max_rows=max_rows)

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

# ===============================================================================

# def build_susy_model(input_dim=18):
#     # Binary classification 0/1 with a single sigmoid output
#     model = keras.Sequential([
#         layers.Dense(128, activation='relu', input_shape=(input_dim,)),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(1, activation='sigmoid')
#     ])

#     model.compile(
#         optimizer='adam',
#         loss='binary_crossentropy',
#         metrics=['accuracy']
#     )

#     model.summary()
#     return model


def build_susy_model(input_dim=18):
    inp = keras.Input(shape=(input_dim,))

    x = layers.BatchNormalization()(inp)

    x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)

    x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.20)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)

    try:
        opt = keras.optimizers.AdamW(learning_rate=2e-3, weight_decay=1e-5)
    except Exception:
        opt = keras.optimizers.Adam(learning_rate=2e-3)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="acc"),
        ],
    )
    
    model.summary()
    return model

# ===============================================================================

def run_susy():                                                                                 
    
    X_train, y_train, X_test, y_test, class_names = load_susy_data(max_rows=1000000, train_size=900000, path="../data/SUSY.csv")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    
    model = build_susy_model(input_dim=X_train.shape[1])

    print("\n[SUSY] Training...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max", factor=0.5, patience=1, min_lr=1e-5
        ),
    ]
    
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=6,  batch_size=4096, verbose=2, 
                        callbacks=callbacks, 
                        shuffle=True)

    print("\n[SUSY] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[SUSY] Test loss: {test_loss:.4f}")
    print(f"[SUSY] Test accuracy: {test_acc:.4f}")

# ======================================================================
# MNIST DATASET

def load_mnist_data():

    print("Loading from tf.keras.datasets.mnist")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0

    print("Train shape:", X_train.shape, "Labels:", y_train.shape)
    print("Test shape:", X_test.shape, "Labels:", y_test.shape)

    class_names = [str(i) for i in range(10)]
    
    return X_train, y_train, X_test, y_test, class_names

# ===============================================================================
# about 784 × 256 = 200,704 weights 

# def build_mnist_model(input_shape=(28, 28), num_classes=10):

#     model = keras.Sequential([
#         layers.Input(shape=input_shape),
#         layers.Reshape((28, 28, 1)),

#         layers.Conv2D(16, 3, padding="same", use_bias=False),
#         layers.BatchNormalization(),
#         layers.Activation("relu"),
#         layers.MaxPooling2D(),

#         layers.Conv2D(32, 3, padding="same", use_bias=False),
#         layers.BatchNormalization(),
#         layers.Activation("relu"),

#         layers.GlobalAveragePooling2D(),
#         layers.Dense(num_classes, activation="softmax"),
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(1e-3),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )
    
#     model.summary()
#     print("Trainable params:", model.count_params())
#     return model

# ===============================================================================

# def build_mnist_model(input_shape=(28, 28), num_classes=10):
#     model = keras.Sequential([
#         layers.Input(shape=input_shape),
#         layers.Reshape((28, 28, 1)),
#         layers.Conv2D(8, 3, padding="same", use_bias=True),
#         layers.Activation("relu"),
#         layers.MaxPooling2D(),  # 28x28 -> 14x14
#         layers.Conv2D(16, 3, padding="same", use_bias=True),
#         layers.Activation("relu"),
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(num_classes, activation="softmax"),
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(1e-3),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     model.summary()
#     print("Trainable params:", model.count_params())
#     return model

# ===============================================================================

# def build_mnist_model(input_shape=(28, 28), num_classes=10, lr=1e-3):

#     model = keras.Sequential([
#         layers.Input(shape=input_shape),
#         layers.Reshape((28, 28, 1)),
#         layers.Conv2D(
#             filters=8,
#             kernel_size=(3, 3),
#             strides=(1, 1),
#             padding="valid",
#             activation="relu",
#             use_bias=True
#         ),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
#         layers.Conv2D(
#             filters=16,
#             kernel_size=(3, 3),
#             strides=(1, 1),
#             padding="valid",
#             activation="relu",
#             use_bias=True
#         ),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
#         layers.Flatten(),
#         layers.Dense(32, activation="tanh", use_bias=True),
#         layers.Dense(num_classes, activation="softmax", use_bias=True),
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=lr),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     model.summary()
#     print("Trainable params:", model.count_params())

#     return model

# ===============================================================================

# def build_mnist_model(input_shape=(28, 28), num_classes=10):

#     model = keras.Sequential([
#         layers.Input(shape=input_shape),
#         layers.Reshape((28, 28, 1)),
#         layers.Conv2D(
#             filters=8,
#             kernel_size=(3, 3),
#             strides=(1, 1),
#             padding="valid",
#             activation="relu",
#             use_bias=True
#         ),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
#         layers.Conv2D(
#             filters=16,
#             kernel_size=(3, 3),
#             strides=(1, 1),
#             padding="valid",
#             activation="relu",
#             use_bias=True
#         ),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
#         layers.Flatten(),
#         layers.Dense(num_classes, activation="softmax", use_bias=True),
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     model.summary()
#     print("Trainable params:", model.count_params())

#     return model

# ===============================================================================
# Best (From Geeks For Geeks):

def build_mnist_model(input_shape=(28, 28), num_classes=10, lr=1e-3): 

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
        ),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"), 
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
        ),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(250, activation="sigmoid"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print("Trainable params:", model.count_params())

    return model

# ===============================================================================

def run_mnist():
    X_train, y_train, X_test, y_test, class_names = load_mnist_data()

    model = build_mnist_model(input_shape=X_train.shape[1:])

    print("\nTraining...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    history = model.fit(X_train,y_train,validation_split=0.1,epochs=5,
                        batch_size=128,verbose=2,callbacks=callbacks)

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


# ======================================================================
# MNIST5 DATASET

def load_mnist5_data(remap_labels=True):

    print("Loading from tf.keras.datasets.mnist")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    train_mask = (y_train >= 0) & (y_train <= 4)
    test_mask  = (y_test  >= 0) & (y_test  <= 4)

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test,  y_test  = X_test[test_mask],  y_test[test_mask]

    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0

    if remap_labels:
        y_train = y_train.astype("int32")
        y_test  = y_test.astype("int32")

    print("MNIST5 Train shape:", X_train.shape, "Labels:", y_train.shape, "classes:", np.unique(y_train))
    print("MNIST5 Test shape:",  X_test.shape,  "Labels:", y_test.shape,  "classes:", np.unique(y_test))

    class_names = [str(i) for i in range(5)]

    return X_train, y_train, X_test, y_test, class_names

# ======================================================================

def build_mnist5_model(input_shape=(28, 28), num_classes=5):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),
        layers.Conv2D(8, 3, padding="same", use_bias=True),
        layers.Activation("relu"),
        layers.MaxPooling2D(),  # 28x28 -> 14x14
        layers.Conv2D(16, 3, padding="same", use_bias=True),
        layers.Activation("relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print("Trainable params:", model.count_params())
    return model

# ======================================================================

# def build_mnist5_model(input_shape=(28, 28), num_classes=4, lr=1e-3):

#     model = keras.Sequential([
#         layers.Input(shape=input_shape),
#         layers.Reshape((28, 28, 1)),
#         layers.Conv2D(
#             filters=8,
#             kernel_size=(3, 3),
#             strides=(1, 1),
#             padding="valid",
#             activation="relu",
#             use_bias=True
#         ),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
#         layers.Conv2D(
#             filters=16,
#             kernel_size=(3, 3),
#             strides=(1, 1),
#             padding="valid",
#             activation="relu",
#             use_bias=True
#         ),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
#         layers.Flatten(),
#         layers.Dense(32, activation="tanh", use_bias=True),
#         layers.Dense(num_classes, activation="softmax", use_bias=True),
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=lr),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     model.summary()
#     print("Trainable params:", model.count_params())

#     return model

# ======================================================================
# Run MNIST5
# ======================================================================

def run_mnist5(epochs=5, batch_size=128, max_train=None, max_test=None):
    
    X_train, y_train, X_test, y_test, class_names = load_mnist5_data()

    model = build_mnist5_model(input_shape=X_train.shape[1:], num_classes=5)

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    return model, history

# ======================================================================
# Pendigits Dataset

def load_pendigits_data(base_path="../data"):

    train_path = os.path.join(base_path, "pendigits.tra")
    test_path  = os.path.join(base_path, "pendigits.tes")

    print(f"Loading from: {base_path}")

    train = np.loadtxt(train_path, delimiter=",", dtype=np.float32)
    test  = np.loadtxt(test_path,  delimiter=",", dtype=np.float32)

    X_train = train[:, :-1].astype(np.float32)     # 16 features
    y_train = train[:, -1].astype(np.int64)        
    X_test  = test[:, :-1].astype(np.float32)
    y_test  = test[:, -1].astype(np.int64)

    class_names = [str(i) for i in range(10)]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    evaluate_dataset(X_train, y_train, X_test, y_test, 10)

    return X_train, y_train, X_test, y_test, class_names

# ===============================================================================

def build_pendigits_model(input_dim=16, num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model

# ===============================================================================

# def build_pendigits_model(input_dim=16, num_classes=10):

#     model = keras.Sequential([
#         layers.Input(shape=(input_dim,)),
#         layers.Dense(num_classes, activation="softmax"),
#     ])

#     model.compile(
#         optimizer=keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     model.summary()
#     return model

# ===============================================================================

def run_pendigits():
    X_train, y_train, X_test, y_test, class_names = load_pendigits_data()

    model = build_pendigits_model(
        input_dim=X_train.shape[1],
        num_classes=len(class_names)
    )

    print("\nTraining...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=15,
        batch_size=128,
        verbose=2,
        shuffle=True
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    save_model_as_flat_txt(model)

# ======================================================================
# CIFAR-10 DATASET

def load_cifar10_data(batch_size: int = 128, buffer_size: int = 50_000):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print(f"x_train shape: {x_train.shape}")    # (50000, 32, 32, 3) => 3 comes from RGB => 15000 ON 3 CLASSES
    print(f"x_test shape: {x_test.shape}")      # (10000, 32, 32, 3)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds

# ======================================================================

# def build_cifar10_model(input_shape=(32, 32, 3), num_classes: int = 10):

#     augment = tf.keras.Sequential(
#         [
#             layers.RandomFlip("horizontal"),
#             layers.RandomTranslation(0.1, 0.1),
#             layers.RandomRotation(0.05),
#         ],
#         name="augment",
#     )

#     def conv_block(x, filters: int):
#         x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation("relu")(x)

#         x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation("relu")(x)

#         x = layers.MaxPooling2D()(x)
#         return x

#     inputs = layers.Input(shape=input_shape)

#     x = inputs

#     x = conv_block(x, 32)
#     x = conv_block(x, 64)
#     x = conv_block(x, 128)

#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dropout(0.3)(x)
#     outputs = layers.Dense(num_classes, activation="softmax")(x)

#     model = models.Model(inputs, outputs)

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model

# ======================================================================

def build_cifar10_model(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 32 -> 16

        layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 16 -> 8

        layers.Flatten(),                 
        layers.Dense(128, activation="tanh"),   
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print("Trainable params:", model.count_params())
    return model

# ======================================================================

# def build_cifar10_model(input_shape=(32, 32, 3), num_classes=10):
#     model = keras.Sequential([
#         layers.Input(shape=input_shape),
#         layers.Flatten(),  # 3072 features
#         layers.Dense(512, activation="tanh"),     
#         layers.Dense(256, activation="tanh"),
#         layers.Dense(num_classes, activation="softmax"),
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     model.summary()
#     print("Trainable params:", model.count_params())
#     return model

# ======================================================================

def run_cifar10(
    batch_size: int = 128,
    epochs: int = 50,
    verbose: int = 1,
):

    train_ds, test_ds = load_cifar10_data(batch_size=batch_size)
    model = build_cifar10_model()
    print("Trainable params:", model.count_params())

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )

    results = model.evaluate(test_ds, verbose=0)
    test_metrics = dict(zip(model.metrics_names, results))
    print("Test metrics:", test_metrics)

    return model, history, test_metrics

# ============================================================
# WINE QUALITY DATASET

def load_wine_type_data(path="../data/winequality.csv", test_size: float = 0.2, 
                        random_state: int = 123):
    print(f"Loading from: {path}")
    df = pd.read_csv(path, sep=",")

    df["type"] = df["type"].map({"white": 0, "red": 1}) 

    feature_cols = [c for c in df.columns if c != "type"]

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=feature_cols + ["type"]).copy()
    after = len(df)
    print(f"Dropped rows with NaNs: {before - after}")

    y = df["type"].astype(np.int32).values
    X = df[feature_cols].astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    num_classes = 2
    return X_train, X_test, y_train, y_test, num_classes

# ============================================================

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

# ============================================================

def run_wine_type():
    X_train, X_test, y_train, y_test, _ = load_wine_type_data(
        path="../data/winequality.csv",
        random_state=123
    )

    evaluate_dataset(X_train, y_train, X_test, y_test, n_classes=2)
    
    model = build_winequality_model(input_dim=X_train.shape[1])

    print("\nTraining...")
    t0 = time.perf_counter()

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        verbose=2,
        callbacks=[
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        ]
    )
    t1 = time.perf_counter()
    print(f"fit() took {t1 - t0:.3f} seconds")
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    return model

# ======================================================================
# Main
# ======================================================================

def main():
    print("DATASET:", DATASET)

    if DATASET == "iris":
        run_iris()
    elif DATASET == "susy":
        run_susy()
    elif DATASET == "mnist":
        run_mnist()
    elif DATASET == "mnist5":
        run_mnist5() 
    elif DATASET == "pendigits":
        run_pendigits()         
    elif DATASET == "cifar10":
        run_cifar10()
    elif DATASET == "winequality":
        run_wine_type()
    else:
        print("DATASET not detected")
        
if __name__ == "__main__":
    main()

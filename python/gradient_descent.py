
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
# python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

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
# Save model weights in flat format
# =======================================================================

def save_all_trainable_as_flat_txt(model, path="model_weights_flat.txt"):
    flat = []
    for var in model.trainable_variables:
        flat.extend(var.numpy().reshape(-1).astype("float32").tolist())

    with open(path, "w") as f:
        for v in flat:
            f.write(f"{v}\n")

    print(f"Saved {len(flat)} trainable params to {path}")
    return flat

# =======================================================================

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


# ============================================================
# IRIS DATASET
# ============================================================

def load_iris_data(
    test_size: float = 0.2,
    random_state: int = 123,
    standardize: bool = True,
):
    print("Loading Iris dataset (scikit-learn)...")

    iris = sk_load_iris()
    X = iris.data.astype(np.float32)          # shape (150, 4)
    y = iris.target.astype(np.int32)          # labels 0,1,2
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
# MODEL (DL4J-equivalent: 4 -> 16 -> 16 -> 3)
# ============================================================

def build_iris_model(
    input_dim: int,
    num_classes: int = 3,
    seed: int = 123,
    lr: float = 1e-3,
):
    # Match DL4J seed behavior (best-effort for TF/Keras)
    tf.keras.utils.set_random_seed(seed)

    inputs = keras.Input(shape=(input_dim,), name="features")

    x = layers.Dense(16, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(16, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="iris_mlp")

    # DL4J MCXENT == multiclass cross-entropy
    # Since y is integer class ids (0/1/2), use SparseCategoricalCrossentropy.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    model.summary()
    print("Total params:", model.count_params())
    return model

# ============================================================
# RUN PIPELINE
# ============================================================

def run_iris(
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 123,
    epochs: int = 40,
    batch_size: int = 16,
):
    # Make an explicit train/val/test split (no validation_split in fit)
    X_train, X_test, y_train, y_test, num_classes, scaler = load_iris_data(
        test_size=test_size,
        random_state=random_state,
        standardize=True,
    )

    # Split train -> train/val
    from sklearn.model_selection import train_test_split
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
# ======================================================================

def load_susy_data(max_rows=1000000, train_size=900000, path="../data/SUSY.csv"):
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

    # Block 1
    x = layers.Dense(512, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)

    # Block 2
    x = layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.20)(x)

    # Head
    out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inp, out)

    # If AdamW is available in your TF version:
    try:
        opt = keras.optimizers.AdamW(learning_rate=2e-3, weight_decay=1e-5)
    except Exception:
        opt = keras.optimizers.Adam(learning_rate=2e-3)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=[
            # keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="acc"),
        ],
    )
    
    model.summary()
    return model

# ===============================================================================

def run_susy():                                                                                 
    
    # X_train, y_train, X_test, y_test, class_names = load_susy_data(max_rows=5000000, train_size=4000000, path="../data/SUSY.csv")
    
    X_train, y_train, X_test, y_test, class_names = load_susy_data(max_rows=1000000, train_size=900000, path="../data/SUSY.csv")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)
    
    model = build_susy_model(input_dim=X_train.shape[1])

    print("\n[SUSY] Training...")
    # callbacks = [
    #     keras.callbacks.EarlyStopping(
    #         monitor="val_auc", mode="max", patience=3, restore_best_weights=True
    #     ),
    #     keras.callbacks.ReduceLROnPlateau(
    #         monitor="val_auc", mode="max", factor=0.5, patience=1, min_lr=1e-5
    #     ),
    # ]
    
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=3,  batch_size=4096, verbose=2, 
                        # callbacks=callbacks, 
                        shuffle=True)

    print("\n[SUSY] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[SUSY] Test loss: {test_loss:.4f}")
    print(f"[SUSY] Test accuracy: {test_acc:.4f}")

    # save_model_as_flat_txt(model, path=f"model_serialization/{DATASET}_model_weights.txt")

# ======================================================================
# BANK DATASET
# ======================================================================

def load_bank_data(path="../data/bank-additional-full.csv"):
    print(f"Loading from: {path}")
    df = pd.read_csv(path, sep=';')

    # Label: yes or no to 1 or 0
    y = (df["y"] == "yes").astype(int).values

    X = pd.get_dummies(df.drop(columns=["y"]), drop_first=True).astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
        # random_state=123 => sudo randomly ordered dataset
        # stratify = y: Split the data so that each class in y appears in the train and test sets 
            # in the same proportion as the original dataset.
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Train shape:", X_train.shape, "Labels:", y_train.shape)
    print("Test  shape:", X_test.shape, "Labels:", y_test.shape)

    return X_train, X_test, y_train, y_test

# ===============================================================================

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

# ===============================================================================

def run_bank():
    X_train, X_test, y_train, y_test = load_bank_data(path="../data/bank-additional-full.csv")

    model = build_bank_model(input_dim=X_train.shape[1])

    print("\nTraining...")
    history = model.fit(X_train, y_train, validation_split=0.2,epochs=3,batch_size=256,verbose=2)

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # save_model_as_flat_txt(model, path=f"model_serialization/{DATASET}_model_weights.txt")

# ======================================================================
# MNIST DATASET
# ======================================================================

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

# def build_mnist_model(input_shape=(28, 28), num_classes=10, lr=1e-3):   # 1e-3 is actually the classic default and usually a good starting point for MNIST/CNNs.

#     model = keras.Sequential([
#         layers.Input(shape=input_shape),
#         layers.Reshape((28, 28, 1)),
#         layers.Conv2D(
#             filters=32,
#             kernel_size=(3, 3),
#             activation="relu",
#         ),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),     # for best accuracy comment out this layer
#         layers.Conv2D(
#             filters=64,
#             kernel_size=(3, 3),
#             activation="relu",
#         ),
#         layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
#         layers.Dropout(0.5),
#         layers.Flatten(),
#         layers.Dense(250, activation="sigmoid"),
#         layers.Dense(num_classes, activation="softmax"),
#     ])

#     # model.compile(
#     #     optimizer=keras.optimizers.Adam(learning_rate=lr),
#     #     loss="sparse_categorical_crossentropy",
#     #     metrics=["accuracy"],
#     # )
#     model.compile(
#         optimizer=keras.optimizers.Adam(),
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
#         layers.Flatten(),                              # 28*28 = 784
#         layers.Dense(32, activation="tanh", use_bias=True),   # hidden layer
#         layers.Dense(num_classes, activation="softmax", use_bias=True),  # output
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(),
#         loss="sparse_categorical_crossentropy",        # integer labels
#         metrics=["accuracy"],
#     )

#     model.summary()
#     print("Trainable params:", model.count_params())
#     return model

# ===============================================================================

# def build_mnist_model(input_shape=(28, 28), num_classes=10):
#     model = keras.Sequential([
#         layers.Input(shape=input_shape),
#         layers.Flatten(),                              # 28*28 = 784
#         layers.Dense(num_classes, activation="softmax", use_bias=True),  # output
#     ])

#     model.compile(
#         optimizer=keras.optimizers.Adam(),
#         loss="sparse_categorical_crossentropy",        # integer labels
#         metrics=["accuracy"],
#     )

#     model.summary()
#     print("Trainable params:", model.count_params())
#     return model
# ===============================================================================

def build_mnist_model(input_shape=(28, 28), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((28, 28, 1)),                    # add channel dim
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),  # 28x28x1 -> 14x14x1
        layers.Flatten(),                               # 14*14 = 196
        layers.Dense(num_classes, activation="softmax", use_bias=True),
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
    # save_all_trainable_as_flat_txt(model, "pre_model_weights_flat.txt")

    history = model.fit(X_train,y_train,validation_split=0.1,epochs=5,
                        batch_size=128,verbose=2,callbacks=callbacks)

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # save_all_trainable_as_flat_txt(model)

# ======================================================================
# MNIST5 DATASET (use only classes 0..3 => classes in total, drop the others)
# ======================================================================

def load_mnist5_data(remap_labels=True):

    print("Loading from tf.keras.datasets.mnist")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    train_mask = (y_train >= 0) & (y_train <= 4)
    test_mask  = (y_test  >= 0) & (y_test  <= 4)

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test,  y_test  = X_test[test_mask],  y_test[test_mask]

    # Normalize
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0

    if remap_labels:
        # For 0..3 this is already correct, but kept for clarity / future changes
        y_train = y_train.astype("int32")
        y_test  = y_test.astype("int32")

    print("MNIST5 Train shape:", X_train.shape, "Labels:", y_train.shape, "classes:", np.unique(y_train))
    print("MNIST5 Test shape:",  X_test.shape,  "Labels:", y_test.shape,  "classes:", np.unique(y_test))

    class_names = [str(i) for i in range(5)]

    return X_train, y_train, X_test, y_test, class_names

# ======================================================================
# SIMPLER (smaller than before) / FASTER CNN for MNIST5

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
    save_all_trainable_as_flat_txt(model)
    return model, history


# ======================================================================
# Adult income DATASET
# ======================================================================

def load_adult_data():
    
    train_path="../data/adult.data"
    test_path="../data/adult.test"
    
    cols = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country","income"
    ]

    df_train = pd.read_csv(train_path, header=None, names=cols,sep=",",engine="python",skipinitialspace=True)
    df_test = pd.read_csv(test_path,header=None,names=cols,sep=",",engine="python",skipinitialspace=True,skiprows=1)

    df_test["income"] = df_test["income"].astype(str).str.replace(".", "", regex=False)

    df_train.replace("?", np.nan, inplace=True)
    df_test.replace("?", np.nan, inplace=True)

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    y_train = (df_train["income"] == ">50K").astype(np.int32).values
    y_test  = (df_test["income"] == ">50K").astype(np.int32).values

    X_train_df = df_train.drop(columns=["income"])
    X_test_df  = df_test.drop(columns=["income"])

    X_train_oh = pd.get_dummies(X_train_df, drop_first=True)
    X_test_oh  = pd.get_dummies(X_test_df, drop_first=True)

    X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_oh).astype(np.float32)
    X_test  = scaler.transform(X_test_oh).astype(np.float32)

    class_names = ["<=50K", ">50K"]

    print("Train shape:", X_train.shape, "Labels:", y_train.shape)
    print("Test  shape:", X_test.shape, "Labels:", y_test.shape)
    print("Class names:", class_names)

    return X_train, y_train, X_test, y_test, class_names

# ======================================================================

def build_adult_model(input_dim):

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model

# ======================================================================

def run_adult():
    X_train, y_train, X_test, y_test, class_names = load_adult_data()
    
    train_path="../data/adult.data"
    test_path="../data/adult.test"
    
    model = build_adult_model(input_dim=X_train.shape[1])

    print("\nTraining...")
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=256,verbose=2)

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    return model

# ======================================================================
# COVERTYPE DATASET
# ======================================================================

def load_covertype_data(path="../data/covertype.csv"):

    label_col = "Cover_Type"
    print(f"Loading from: {path}")

    df = pd.read_csv(path)
    print("Raw shape:", df.shape)

    y_all = df[label_col].astype(np.int64).to_numpy()      # 1..7
    X_all = df.drop(columns=[label_col]).astype(np.float32).to_numpy()

    y_all = y_all - 1

    rng = np.random.default_rng(123)
    idx = rng.permutation(len(y_all))
    X_all = X_all[idx]
    y_all = y_all[idx]

    n = X_all.shape[0]
    train_size = int(0.8 * n)

    X_train_raw = X_all[:train_size]
    y_train = y_all[:train_size]
    X_test_raw = X_all[train_size:]
    y_test = y_all[train_size:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test  = scaler.transform(X_test_raw).astype(np.float32)

    class_names = [str(i) for i in range(7)]

    print("Train shape:", X_train.shape, "Labels:", y_train.shape, "y range:", (y_train.min(), y_train.max()))
    print("Test  shape:", X_test.shape,  "Labels:", y_test.shape,  "y range:", (y_test.min(), y_test.max()))
    print("Classes:", class_names)

    print("Train label counts:", np.bincount(y_train, minlength=7))
    print("Test  label counts:", np.bincount(y_test, minlength=7))

    return X_train, y_train, X_test, y_test, class_names

# ===============================================================================

def build_covertype_model(input_dim, num_classes=7):
 
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model

# ===============================================================================

def run_covertype():
    
    X_train, y_train, X_test, y_test, class_names = load_covertype_data(path="../data/covertype.csv")

    model = build_covertype_model(input_dim=X_train.shape[1], num_classes=len(class_names))

    print("\ndfTraining...")
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=5, batch_size=256, verbose=2)

    print("\ndf Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"dfTest loss: {test_loss:.4f}")
    print(f"dfTest accuracy: {test_acc:.4f}")

    return model

# ======================================================================
# HAR DATASET (Human Activity Recognition)
# ======================================================================

def load_har_data(base_path="../data/"):

    train_x_path = os.path.join(base_path, "X_train.txt")
    train_y_path = os.path.join(base_path, "y_train.txt")
    test_x_path  = os.path.join(base_path, "X_test.txt")
    test_y_path  = os.path.join(base_path, "y_test.txt")

    print(f"[HAR] Loading from: {base_path}")
    X_train = np.loadtxt(train_x_path, dtype=np.float32)
    y_train = np.loadtxt(train_y_path, dtype=np.int64)
    X_test  = np.loadtxt(test_x_path,  dtype=np.float32)
    y_test  = np.loadtxt(test_y_path,  dtype=np.int64)

    y_train = y_train - 1   # 1 ... 6 -> 0 ... 5
    y_test  = y_test - 1

    class_names = [str(i) for i in range(6)]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    return X_train, y_train, X_test, y_test, class_names

# ===============================================================================

def build_har_model(input_dim=561, num_classes=6):

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model

# ===============================================================================

def run_har():
    X_train, y_train, X_test, y_test, class_names = load_har_data(
        base_path="../data/",
        normalize=True
    )

    model = build_har_model(input_dim=X_train.shape[1], num_classes=len(class_names))

    print("\n[HAR] Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=256,
        verbose=2
    )

    print("\n[HAR] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[HAR] Test loss: {test_loss:.4f}")
    print(f"[HAR] Test accuracy: {test_acc:.4f}")


# ======================================================================
# Pen Digits Dataset
# ======================================================================

def load_pendigits_data(base_path="../data"):

    train_path = os.path.join(base_path, "pendigits.tra")
    test_path  = os.path.join(base_path, "pendigits.tes")

    print(f"Loading from: {base_path}")

    train = np.loadtxt(train_path, delimiter=",", dtype=np.float32)
    test  = np.loadtxt(test_path,  delimiter=",", dtype=np.float32)

    # last column is label
    X_train = train[:, :-1].astype(np.float32)     # (n, 16)
    y_train = train[:, -1].astype(np.int64)        # (n,)
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
# Pen Digits Half
# ======================================================================

def load_pendigits_half_data(base_path="../data"):

    train_path = os.path.join(base_path, "pendigits.tra")
    test_path  = os.path.join(base_path, "pendigits.tes")

    print(f"Loading from: {base_path}")

    train = np.loadtxt(train_path, delimiter=",", dtype=np.float32)
    test  = np.loadtxt(test_path,  delimiter=",", dtype=np.float32)

    # Split features / labels
    X_train = train[:, :-1].astype(np.float32)
    y_train = train[:, -1].astype(np.int64)

    X_test  = test[:, :-1].astype(np.float32)
    y_test  = test[:, -1].astype(np.int64)

    # ----------------------------------------------------
    # KEEP ONLY DIGITS 0–4
    # ----------------------------------------------------
    train_mask = y_train < 5
    test_mask  = y_test < 5

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    class_names = [str(i) for i in range(5)]

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    evaluate_dataset(X_train, y_train, X_test, y_test, 5)

    return X_train, y_train, X_test, y_test, class_names

# ===============================================================================

def build_pendigits_half_model(input_dim=16, num_classes=5):

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

def run_pendigits_half():
    X_train, y_train, X_test, y_test, class_names = load_pendigits_half_data()

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

# ======================================================================
# CIFAR-10 DATASET
# ======================================================================

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

        # Block 1
        layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 32 -> 16

        # Block 2
        layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 16 -> 8

        # Classifier head (MNIST-style)
        layers.Flatten(),                        # 8*8*64 = 4096
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
#         layers.Dense(512, activation="tanh"),          # PSO-friendly (smooth)
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

# =======================================================================================================
# CIFAR3
# =======================================================================================================

def load_cifar3_data(
    classes=(0, 1, 2),
    batch_size: int = 128,
    buffer_size: int = 50_000,
):
    # CIFAR-10 labels are: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Flatten labels from shape (N,1) -> (N,)
    y_train = y_train.squeeze().astype(np.int64)
    y_test = y_test.squeeze().astype(np.int64)

    # Filter to selected classes
    classes = np.array(classes, dtype=np.int64)

    train_mask = np.isin(y_train, classes)
    test_mask = np.isin(y_test, classes)

    x_train = x_train[train_mask].astype("float32") / 255.0
    y_train = y_train[train_mask]
    x_test = x_test[test_mask].astype("float32") / 255.0
    y_test = y_test[test_mask]

    # Remap labels to 0..(K-1) so sparse_categorical_crossentropy works cleanly
    remap = {int(c): i for i, c in enumerate(classes.tolist())}
    y_train = np.vectorize(remap.get)(y_train).astype(np.int64)
    y_test = np.vectorize(remap.get)(y_test).astype(np.int64)

    cifar10_names = ["airplane", "automobile", "bird", "cat", "deer",
                     "dog", "frog", "horse", "ship", "truck"]
    class_names = [cifar10_names[int(c)] for c in classes]

    print(f"Selected CIFAR-3 classes: {classes.tolist()} -> {class_names}")
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test : {x_test.shape},  y_test : {y_test.shape}")

    # Build tf.data pipelines
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(min(buffer_size, len(x_train)))
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, test_ds, class_names

# =======================================================================================================

# def build_cifar3_model(input_shape=(32, 32, 3), num_classes: int = 3):

#     inputs = layers.Input(shape=input_shape)

#     x = layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
#     x = layers.MaxPooling2D()(x)

#     x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
#     x = layers.MaxPooling2D()(x)

#     x = layers.GlobalAveragePooling2D()(x)
#     outputs = layers.Dense(num_classes, activation="softmax")(x)

#     model = models.Model(inputs, outputs)

#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )

#     model.summary()
#     print("Trainable params:", model.count_params())

#     return model


def build_cifar3_model(input_shape=(32, 32, 3), num_classes: int = 3):

    inputs = layers.Input(shape=input_shape)

    # DL4J: padding(0,0) => "valid" in Keras (no padding)
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding="valid", activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="valid", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="valid", activation="relu")(x)

    # DL4J DenseLayer implicitly flattens conv output
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    print("Trainable params:", model.count_params())

    return model

# =======================================================================================================

def run_cifar3(
    classes=(0, 1, 2),
    batch_size: int = 256,
    epochs: int = 30,
    verbose: int = 1,
):

    print("GPUs:", tf.config.list_physical_devices("GPU"))

    train_ds, test_ds, class_names = load_cifar3_data(
        classes=classes,
        batch_size=batch_size,
    )

    model = build_cifar3_model(num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
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

    return model, history, test_metrics, class_names


# =======================================================================================================
# HIGGS
# =======================================================================================================

def load_higgs_data(
    path: str = "../data/higgs/HIGGS.csv.gz",
    nrows: int = 620_000,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 123,
):
    print(f"Loading HIGGS from: {path}")

    df = pd.read_csv(path, sep=",", header=None, nrows=nrows)

    y = df.iloc[:, 0].astype(np.int32).values
    X = df.iloc[:, 1:].astype(np.float32).values

    # train / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # train / val
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac,
        random_state=random_state,
        stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# =======================================================================================================

def build_higgs_model(
        
    input_dim: int = 29,
    hidden1: int = 1024,
    hidden2: int = 512,
    hidden3: int = 128,
    dropout: float = 0.15,
    lr: float = 1e-3,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(hidden1, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(hidden2, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(hidden3, activation="relu")(x)
    logits = tf.keras.layers.Dense(1, name="logits")(x)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name="higgs_dense")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.0),
            tf.keras.metrics.AUC(name="auc", from_logits=True),
        ],
    )
    return model


# =======================================================================================================

def run_higgs():

    X_train, y_train, X_val, y_val, X_test, y_test = load_higgs_data()

    model = build_higgs_model(input_dim=X_train.shape[1])

    print(model.summary())

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=4096,
        shuffle=True,
        verbose=1
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc:  {test_acc:.4f}")
    print(f"Test AUC:  {test_auc:.4f}")

    return model


# ============================================================
# WINE QUALITY DATASET
# ============================================================


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
    return X_train, X_test, y_train, y_test, num_classes

# ============================================================
# MODEL
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
# RUN PIPELINE
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


# ============================================================
# LETTER
# ============================================================

def load_letter_recognition_data(csv_path="../data/letter-recognition.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop("letter", axis=1).values
    y = df["letter"].values
    # Encode letters to integers
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    return train_test_split(X, y_enc, test_size=0.2, random_state=42)

# def build_letter_recognition_model(input_dim, num_classes):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
#     model.compile(
#         optimizer='adam',
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     return model

def build_letter_recognition_model(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def evaluate_letter_recognition():
    X_train, X_test, y_train, y_test = load_letter_recognition_data()
    model = build_letter_recognition_model(X_train.shape[1], 26)
    history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.1)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", test_acc)
    # save_model_as_flat_txt(model)

def run_letter():

    evaluate_letter_recognition()

# ======================================================================
# Main
# ======================================================================

def main():
    print("DATASET:", DATASET)

    if DATASET == "iris":
        run_iris()
    elif DATASET == "susy":
        run_susy()
    elif DATASET == "bank":
        run_bank()
    elif DATASET == "mnist":
        run_mnist()
    elif DATASET == "mnist5":
        run_mnist5()
    elif DATASET == "adult":
        run_adult()
    elif DATASET == "covertype":
        run_covertype()
    elif DATASET == "har":
        run_har()  
    elif DATASET == "pendigits":
        run_pendigits()      
    elif DATASET == "pendigits-half":
        run_pendigits_half()      
    elif DATASET == "cifar10":
        run_cifar10()
    elif DATASET == "cifar3":
        run_cifar3()
    elif DATASET == "higgs":
        run_higgs()
    elif DATASET == "winequality":
        run_wine_type()
    elif DATASET == "letter":
        run_letter()
    else:
        print("DATASET not detected")
        
if __name__ == "__main__":
    main()

# Tensorflow functions:
# load_{dataset_name}_data()
# build_{dataset_name}_model()
# run_{dataset_name}()
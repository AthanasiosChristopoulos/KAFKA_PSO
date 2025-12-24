
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

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

    save_model_as_flat_txt(model, path=f"model_serialization/{DATASET}_model_weights.txt")

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

    save_model_as_flat_txt(model, path=f"model_serialization/{DATASET}_model_weights.txt")

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

def build_mnist_model(input_shape=(28, 28)):
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model

# ===============================================================================

def run_mnist():
    X_train, y_train, X_test, y_test, class_names = load_mnist_data()

    model = build_mnist_model(input_shape=X_train.shape[1:])

    print("\nTraining...")
    history = model.fit(X_train,y_train,validation_split=0.1,epochs=5,batch_size=128,verbose=2,)

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    save_model_as_flat_txt(model, path=f"model_serialization/{DATASET}_model_weights.txt")

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

    print("Train shape:", X_train.shape, "Labels:", y_train.shape,
          "y range:", (int(y_train.min()), int(y_train.max())),
          "counts:", np.bincount(y_train, minlength=10))
    print("Test  shape:", X_test.shape,  "Labels:", y_test.shape,
          "y range:", (int(y_test.min()), int(y_test.max())),
          "counts:", np.bincount(y_test, minlength=10))
    print("Classes:", class_names)

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

# ======================================================================
# CIFAR-10 DATASET
# ======================================================================

def load_cifar10_data(normalize=True, one_hot=False):
    """
    Returns:
      X_train: (50000, 32, 32, 3) float32
      y_train: (50000,) int64   OR (50000,10) if one_hot=True
      X_test : (10000, 32, 32, 3) float32
      y_test : (10000,) int64   OR (10000,10) if one_hot=True
      class_names: list[str] length 10
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    y_train = y_train.squeeze().astype(np.int64)
    y_test  = y_test.squeeze().astype(np.int64)

    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    if normalize:
        X_train /= 255.0
        X_test  /= 255.0

    if one_hot:
        y_train = keras.utils.to_categorical(y_train, 10).astype(np.float32)
        y_test  = keras.utils.to_categorical(y_test, 10).astype(np.float32)

    class_names = [
        "airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"
    ]

    print("[CIFAR-10] Train shape:", X_train.shape, "Labels:", y_train.shape)
    print("[CIFAR-10] Test  shape:", X_test.shape,  "Labels:", y_test.shape)
    print("[CIFAR-10] Classes:", class_names)

    return X_train, y_train, X_test, y_test, class_names

# =======================================================================================================

def build_cifar10_model(input_shape=(32, 32, 3), num_classes=10, weight_decay=1e-4):

    def conv_bn_relu(x, filters, kernel_size=3, strides=1):
        x = layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same",
            use_bias=False, kernel_regularizer=keras.regularizers.l2(weight_decay)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    def residual_block(x, filters, downsample=False):
        strides = 2 if downsample else 1
        shortcut = x

        # First conv
        y = layers.Conv2D(
            filters, 3, strides=strides, padding="same",
            use_bias=False, kernel_regularizer=keras.regularizers.l2(weight_decay)
        )(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation("relu")(y)

        # Second conv
        y = layers.Conv2D(
            filters, 3, strides=1, padding="same",
            use_bias=False, kernel_regularizer=keras.regularizers.l2(weight_decay)
        )(y)
        y = layers.BatchNormalization()(y)

        # Match shortcut shape if needed
        if downsample or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(
                filters, 1, strides=strides, padding="same",
                use_bias=False, kernel_regularizer=keras.regularizers.l2(weight_decay)
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        out = layers.Add()([shortcut, y])
        out = layers.Activation("relu")(out)
        return out

    inp = keras.Input(shape=input_shape)

    # Stem
    x = conv_bn_relu(inp, 16, 3, 1)

    # Stage 1: 16 filters, 3 blocks
    for _ in range(3):
        x = residual_block(x, 16, downsample=False)

    # Stage 2: 32 filters, 3 blocks (first downsample)
    x = residual_block(x, 32, downsample=True)
    for _ in range(2):
        x = residual_block(x, 32, downsample=False)

    # Stage 3: 64 filters, 3 blocks (first downsample)
    x = residual_block(x, 64, downsample=True)
    for _ in range(2):
        x = residual_block(x, 64, downsample=False)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    )(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inp, out)

    # Optimizer: Adam is fine; SGD+momentum often edges higher for CIFAR.
    # We'll use SGD+Nesterov for a classic reliable CIFAR setup.
    opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model

# =======================================================================================================

def run_cifar10(epochs=50, batch_size=128, use_augmentation=True):
    X_train, y_train, X_test, y_test, class_names = load_cifar10_data(
        normalize=True,
        one_hot=False
    )

    model = build_cifar10_model(input_shape=X_train.shape[1:], num_classes=10)

    # Data augmentation (standard CIFAR-ish): pad+random crop + flip
    if use_augmentation:
        aug = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.ZeroPadding2D(padding=4),
            layers.RandomCrop(32, 32),
        ])
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(50000).batch(batch_size).map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Learning-rate schedule (simple step-down)
    def lr_schedule(epoch, lr):
        # classic CIFAR schedule: drop at 50% and 75% of training
        if epoch == int(epochs * 0.5) or epoch == int(epochs * 0.75):
            return lr * 0.1
        return lr

    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
    ]

    print("\n[CIFAR-10] Training...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks
    )

    print("\n[CIFAR-10] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[CIFAR-10] Test loss: {test_loss:.4f}")
    print(f"[CIFAR-10] Test accuracy: {test_acc:.4f}")

    return model, history, class_names

# =======================================================================================================
# HIGGS
# =======================================================================================================

AUTOTUNE = tf.data.AUTOTUNE

def load_higgs_data(
    path: str,
    *,
    batch_size: int = 4096,
    train_rows: int = 300_000,
    val_rows: int = 50_000,
    test_rows: int = 50_000,
    shuffle_buffer: int = 200_000,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Download 'HIGGS.csv.gz' from the UCI HIGGS dataset page and pass the local path."
        )

    input_dim = 29
    total_take = train_rows + val_rows + test_rows

    # Define column defaults for CSV parsing:
    # 1 label + 29 features
    record_defaults = [tf.constant(0, dtype=tf.int32)] + [tf.constant(0.0, dtype=tf.float32)] * input_dim

    def _parse_line(*cols):
        # cols is a list/tup of tensors (label + features)
        y = tf.cast(cols[0], tf.int32)
        x = tf.stack([tf.cast(c, tf.float32) for c in cols[1:]], axis=0)  # (29,)
        return x, y

    # Works with .gz as well; tf.data can read compressed text.
    ds = tf.data.experimental.CsvDataset(
        filenames=[path],
        record_defaults=record_defaults,
        header=False,
        field_delim=",",
        use_quote_delim=True,
    )

    ds = ds.map(_parse_line, num_parallel_calls=AUTOTUNE).take(total_take)

    # Split sequentially (simple + deterministic)
    train_ds = ds.take(train_rows)
    remainder = ds.skip(train_rows)
    val_ds = remainder.take(val_rows)
    test_ds = remainder.skip(val_rows).take(test_rows)

    # Shuffle only training
    train_ds = train_ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    # Batch + prefetch
    train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, input_dim

# =======================================================================================================

def build_higgs_model(
    input_dim: int = 29,
    *,
    hidden1: int = 1024,
    hidden2: int = 512,
    hidden3: int = 128,
    dropout: float = 0.15,
    lr: float = 1e-3,
) -> tf.keras.Model:
    """
    Dense model with ~700k parameters by default:
      29 -> 1024 -> 512 -> 128 -> 1

    Param count approx:
      29*1024 + 1024
    + 1024*512 + 512
    + 512*128 + 128
    + 128*1 + 1
    ≈ 30k + 525k + 65k + 129 ≈ 620k (plus biases) => within 100k..1M.
    """
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
            tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.0),  # threshold on logits==0 <=> prob 0.5
            tf.keras.metrics.AUC(name="auc", from_logits=True),
        ],
    )
    return model

# =======================================================================================================

def run_higgs(
    dataset_path: str = "../data/higgs/HIGGS.csv",
    batch_size: int = 4096,
    train_rows: int = 300_000,
    val_rows: int = 50_000,
    test_rows: int = 50_000,
    epochs: int = 10,
):

    train_ds, val_ds, test_ds, input_dim = load_higgs_data(
        dataset_path,
        batch_size=batch_size,
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
    )

    model = build_higgs_model(input_dim=input_dim)

    print(model.summary())
    print(f"Total parameters: {model.count_params():,}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", patience=3, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=2, mode="max", min_lr=1e-5),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    results = model.evaluate(test_ds, verbose=1)
    print("\nTest metrics:", dict(zip(model.metrics_names, results)))

    return model

# ======================================================================
# Main
# ======================================================================

def main():
    print("DATASET:", DATASET)

    if DATASET == "susy":
        run_susy()
    elif DATASET == "bank":
        run_bank()
    elif DATASET == "mnist":
        run_mnist()
    elif DATASET == "adult":
        run_adult()
    elif DATASET == "covertype":
        run_covertype()
    elif DATASET == "har":
        run_har()  
    elif DATASET == "pendigits":
        run_pendigits()      
    elif DATASET == "cifar10":
        run_cifar10()
    elif DATASET == "higgs":
        run_higgs()
    else:
        print("DATASET not detected")
        
if __name__ == "__main__":
    main()

# Tensorflow functions:
# load_{dataset_name}_data()
# build_{dataset_name}_model()
# run_{dataset_name}()
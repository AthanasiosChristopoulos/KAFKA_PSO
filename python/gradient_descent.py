
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
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=3,  batch_size=256, verbose=2)

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
    else:
        print("DATASET not detected")
        
if __name__ == "__main__":
    main()


# load_{dataset_name}_data()
# build_{dataset_name}_model()
# run_{dataset_name}()
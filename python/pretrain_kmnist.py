
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import gzip


DATASET = "KMNIST"

# =====================================================================================

def _read_idx_images_gz(path):
    with gzip.open(path, "rb") as f:
        data = f.read()

    magic = int.from_bytes(data[0:4], "big")
    if magic != 2051:
        raise ValueError(f"Invalid image file magic number in {path}: {magic}")

    num_images = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")

    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(num_images, rows, cols)
    return images

# =======================================================================================

def _read_idx_labels_gz(path):
    with gzip.open(path, "rb") as f:
        data = f.read()

    magic = int.from_bytes(data[0:4], "big")
    if magic != 2049:
        raise ValueError(f"Invalid label file magic number in {path}: {magic}")

    num_labels = int.from_bytes(data[4:8], "big")
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    labels = labels.reshape(num_labels)
    return labels

# =======================================================================================

def load_kmnist(data_dir):
    """
    Load KMNIST from local MNIST-format .gz files stored on NAS.

    Expected files inside data_dir:
      - train-images-idx3-ubyte.gz
      - train-labels-idx1-ubyte.gz
      - t10k-images-idx3-ubyte.gz
      - t10k-labels-idx1-ubyte.gz
    """

    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_images_path  = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path  = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    required_files = [
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
    ]

    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "KMNIST files not found.\n"
            "Please place these files in:\n"
            f"  {data_dir}\n\n"
            "Missing:\n  " + "\n  ".join(missing)
        )

    x_train = _read_idx_images_gz(train_images_path)
    y_train = _read_idx_labels_gz(train_labels_path)
    x_test  = _read_idx_images_gz(test_images_path)
    y_test  = _read_idx_labels_gz(test_labels_path)

    # normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # add channel dim -> (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test  = np.expand_dims(x_test, axis=-1)

    y_train = y_train.astype("int32")
    y_test  = y_test.astype("int32")

    print("KMNIST loaded from local files:")
    print("  x_train:", x_train.shape, x_train.dtype)
    print("  y_train:", y_train.shape, y_train.dtype)
    print("  x_test :", x_test.shape, x_test.dtype)
    print("  y_test :", y_test.shape, y_test.dtype)

    return x_train, y_train, x_test, y_test


# =====================================================================================

def build_kmnist_base_plus_head_v1(input_shape=(28, 28, 1), num_classes=10):

    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),

            # ---- Backbone / feature extractor ----
            layers.Conv2D(16, kernel_size=3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.10),

            layers.Conv2D(32, kernel_size=3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=2),
            layers.Dropout(0.15),

            layers.Conv2D(64, kernel_size=3, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),

            # Instead of Flatten -> keep head tiny
            layers.GlobalAveragePooling2D(name="features"),

            # ---- Tiny classifier head ----
            # Params = (64 + 1) * 10 = 650
            layers.Dense(num_classes, activation="softmax", name="classifier"),
        ],
        name="kmnist_base_plus_head_v1"
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# =====================================================================================

def train_and_export(out_dir="pretrained_model", epochs=10, batch_size=128):

    version = "v1"

    model_registry = {
        "v1": ("kmnist_base_plus_head_v1", build_kmnist_base_plus_head_v1),
    }
    
    filename, mnist_model_function = model_registry[version]

    x_train, y_train, x_test, y_test = load_kmnist(
        data_dir="/mnt/nas_drive/achristopoulos/KAFKA_PSO_4/data/kmnist")

    model = mnist_model_function(input_shape=x_train.shape[1:], num_classes=10)
    name_h5_file = filename
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,   # x_train_full (60000 samples),
                                # 90% training set (54000), 10% → validation set (6000)
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
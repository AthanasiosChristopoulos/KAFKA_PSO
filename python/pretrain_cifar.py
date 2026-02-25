import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from contextlib import contextmanager

DATASET = "cifar10"

@contextmanager
def tee_output(log_path: str):
    """
    Mirror everything printed to stdout/stderr to a log file, while still showing it in the terminal.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    with open(log_path, "w", buffering=1) as f:  # line-buffered
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(old_out, f)
        sys.stderr = Tee(old_err, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

# ================================================================================================

def export_mobilenetv2_base(save_path="pretrained_model/mobilenetv2_base_32x32.h5"):
    # Feature extractor only (no classifier head)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,      # <- important
        weights="imagenet"
    )

    # Freeze it (feature extractor mode)
    base_model.trainable = False

    _ = base_model(tf.zeros([1, 32, 32, 3]), training=False)

    print("\n=== Base model summary ===")
    base_model.summary()

    base_model.save(save_path)
    print(f"\nSaved base model to: {save_path}")

    print("\nLast 20 layer names (for DL4J setFeatureExtractor / removing vertices):")
    for layer in base_model.layers[-20:]:
        print("  ", layer.name)

# ============================================================================================

def export_mobilenetv3small_base(save_path="pretrained_model/mobilenetv3small_32x32.h5"):

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    base_model.summary()

    base_model.save(save_path)


# ===============================================================================
# Load Data

def load_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # y is shape (N,1) -> make it (N,)
    y_train = y_train.astype("int64").reshape(-1)
    y_test  = y_test.astype("int64").reshape(-1)

    # Normalize to [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # shapes: (N, 32, 32, 3)
    return x_train, y_train, x_test, y_test

# ===============================================================================
# Model (Simple CIFAR feature extractor + head)

def build_cifar_base(input_shape=(32, 32, 3), num_classes=10):    # this means NHWC (look at the order in shape input_shape=(32, 32, 3))
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   # 32 -> 16

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   # 16 -> 8

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

def build_cifar_base_v2(input_shape=(32, 32, 3), num_classes=10, feat_dim=64):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 32->16

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 16->8

        layers.Conv2D(feat_dim, 3, padding="same", activation="relu", use_bias=True),

        layers.GlobalAveragePooling2D(),          # -> (feat_dim,)
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

def build_cifar_base_v3(input_shape=(32, 32, 3), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 32->16

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 16->8

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 8->4

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 4->2

        layers.Flatten(),        # 2*2*64 = 256
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

def build_cifar_base_v4(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # 32x32
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 32 -> 16

        # 16x16
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 16 -> 8

        # 8x8
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),
        layers.GlobalAveragePooling2D(),  # -> (128,)

        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

    # Paramater calculations:
    # (3 * 3 * 1 + 1) * 32 = 896
    # (3 * 3 * 32 + 1) * 32 = 9,248
    # (3 * 3 * 32 + 1) * 64 = 18,496
    # (3⋅3⋅64+1)⋅64 = 36,928    
    # (3⋅3⋅64+1)⋅128 = 73,856
    # (3⋅3⋅128+1)⋅128 = 147,584
    # (128+1)⋅10 = 1290, Not dependend from the image dimensionality because of the GlobalPooling Layer
    # Total​=896+9,248+18,496+36,928+73,856+147,584+1,290=288,298​​
    
# ===============================================================================

def train_and_export(out_dir="pretrained_model", epochs=30, batch_size=5000):
    
    x_train, y_train, x_test, y_test = load_cifar10()

    version = "v4"

    if(version == "v1"):
        model = build_cifar_base(input_shape=x_train.shape[1:], num_classes=10)
        name_h5_file = "cifar10_base_plus_head_v1"
    elif(version == "v2"):
        model = build_cifar_base_v2(input_shape=x_train.shape[1:], num_classes=10)
        name_h5_file = "cifar10_base_plus_head_v2"
    elif(version == "v3"):
        model = build_cifar_base_v3(input_shape=x_train.shape[1:], num_classes=10)
        name_h5_file = "cifar10_base_plus_head_v3"

    elif(version == "v4"):
        model = build_cifar_base_v4(input_shape=x_train.shape[1:], num_classes=10)
        name_h5_file = "cifar10_base_plus_head_v4"

    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, f"{name_h5_file}.h5")
    log_path = os.path.join(out_dir, f"{name_h5_file}.txt")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    # Everything printed inside here goes both to terminal AND to the .log file
    with tee_output(log_path):
        print(f"Logging training output to: {log_path}")
        print(f"Will save model to: {h5_path}\n")

        history = model.fit(
            x_train, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            callbacks=callbacks
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"\nCIFAR-10 test acc: {test_acc:.4f}, loss: {test_loss:.4f}")

        model.save(h5_path)
        print("Saved Keras H5:", h5_path)

    return model, history

# ===============================================================================

if __name__ == "__main__":
    
    # export_mobilenetv2_base()
    # export_mobilenetv3small_base()
    train_and_export()


    # source ~/venvs/tf215/bin/activate

# Ranking of overfiiting highest to lowest:
# build_cifar_base_v3 (Flatten + Dense)
# build_cifar_base (v1) (heavier conv stack + 128 features head)
# build_cifar_base_v4 (heavier conv stack but GAP helps)
# build_cifar_base_v2 (smallest; GAP; lowest capacity
# MobileNetV2 / MobileNetV3Small (frozen base) => this provides with the least overfitting
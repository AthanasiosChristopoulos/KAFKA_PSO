import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from contextlib import contextmanager
from pathlib import Path
import shutil


DATASET = "cifar10"
CINIC_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]
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


# ================================================================================================

def export_mobilenetv2_base_224(save_path="pretrained_model/mobilenetv2_base_224x224.h5"):
    # Feature extractor only (no classifier head)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")

    base_model.trainable = False

    _ = base_model(tf.zeros([1, 224, 224, 3]), training=False)

    print("\n=== Base model summary ===")
    base_model.summary()

    base_model.save(save_path)
    print(f"\nSaved base model to: {save_path}")

    print("\nLast 20 layer names (for DL4J setFeatureExtractor / removing vertices):")
    for layer in base_model.layers[-20:]:
        print("  ", layer.name)

# ============================================================================================

def export_mobilenet_base_224(save_path="pretrained_model/mobilenet_base_224x224.h5"):
    # Feature extractor only (no classifier head)
    base_model = tf.keras.applications.MobileNet(input_shape=(224,224,3), include_top=False, weights="imagenet")

    base_model.trainable = False

    _ = base_model(tf.zeros([1, 224, 224, 3]), training=False)

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

def load_cifar100():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode="fine")

    y_train = y_train.astype("int64").reshape(-1)
    y_test  = y_test.astype("int64").reshape(-1)

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    return x_train, y_train, x_test, y_test

# ===============================================================================

def load_cinic10(
    root_dir: str,
    img_size=(32, 32),
    batch_size: int = 256,
    seed: int = 42,
    use_valid_as_test: bool = False,
):
    train_dir = os.path.join(root_dir, "train")
    valid_dir = os.path.join(root_dir, "valid")
    test_dir  = os.path.join(root_dir, "test")

    if not (os.path.isdir(train_dir) and os.path.isdir(valid_dir) and os.path.isdir(test_dir)):
        raise FileNotFoundError(
            f"Expected CINIC-10 dirs at:\n"
            f"  {train_dir}\n  {valid_dir}\n  {test_dir}\n"
            f"(Did you extract CINIC-10 correctly?)"
        )

    def make_ds(path, shuffle):
        return keras.utils.image_dataset_from_directory(
            path,
            labels="inferred",
            label_mode="int",          # integer labels
            class_names=CINIC_CLASSES, # enforce CIFAR-10 ordering
            image_size=img_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    train_ds = make_ds(train_dir, shuffle=True)
    val_ds   = make_ds(valid_dir, shuffle=False)

    if use_valid_as_test:
        test_ds = val_ds
    else:
        test_ds  = make_ds(test_dir, shuffle=False)

    # Normalize to [0,1]
    def norm(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        y = tf.cast(y, tf.int64)
        return x, y

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(norm, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(norm,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds  = test_ds.map(norm,  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds

# ===============================================================================

def _prepare_tinyimagenet_val_folders(tiny_root: str) -> str:

    tiny_root = Path(tiny_root)
    val_dir = tiny_root / "val"
    images_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"

    if not ann_path.exists():
        raise FileNotFoundError(f"Missing {ann_path}")

    out_dir = tiny_root / "val_prepared"
    # If already prepared, reuse it
    if out_dir.exists() and any(out_dir.iterdir()):
        return str(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse annotations: each line: <img> <wnid> <x1> <y1> <x2> <y2>
    mapping = {}
    with ann_path.open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_name, wnid = parts[0], parts[1]
                mapping[img_name] = wnid

    # Copy images into class folders
    for img_name, wnid in mapping.items():
        src = images_dir / img_name
        if not src.exists():
            continue
        dst_dir = out_dir / wnid
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / img_name
        if not dst.exists():
            shutil.copy2(src, dst)

    return str(out_dir)

# ===============================================================================

def load_tiny_imagenet200(tiny_root: str, batch_size: int = 128, img_size=(64, 64), seed: int = 1337):

    tiny_root = Path(tiny_root)
    train_dir = tiny_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train dir: {train_dir}")

    val_prepared_dir = _prepare_tinyimagenet_val_folders(str(tiny_root))

    train_ds = keras.utils.image_dataset_from_directory(
        str(train_dir),
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        val_prepared_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
    )

    # Good practice: normalize + cache/prefetch
    norm = layers.Rescaling(1.0 / 255.0)

    def _norm_map(x, y):
        return norm(x), y

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(_norm_map, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.map(_norm_map, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds

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

def build_cifar_base_v5(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Base CNN (feature extractor) - CIFAR-ready
        # 32x32
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 32 -> 16

        # 16x16
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 16 -> 8

        # 8x8
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True),

        # REQUIRED tail (exact pattern like your MNIST model)
        layers.Flatten(),

        # Optional base representation layer
        layers.Dense(64, activation="relu", use_bias=True),

        # Head for CIFAR pretraining / training
        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ===============================================================================

def build_cifar_base_v5(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Base CNN (feature extractor) - CIFAR-ready
        # 32x32
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 32 -> 16

        # 16x16
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 16 -> 8

        # 8x8
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True),

        # REQUIRED tail (exact pattern like your MNIST model)
        layers.Flatten(),

        # Optional base representation layer
        layers.Dense(64, activation="relu", use_bias=True),

        # Head for CIFAR pretraining / training
        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ===============================================================================

def build_cifar_base_v6(input_shape=(32, 32, 3), num_classes=10):
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
        layers.Conv2D(96, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(96, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 8 -> 4

        # 4x4
        layers.Conv2D(96, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 4 -> 2

        # REQUIRED tail style: Flatten -> Dense(num_classes)
        layers.Flatten(),

        # FINAL head (params < 5000 guaranteed if num_classes=10)
        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ===============================================================================

def build_cinic_base_v1(input_shape=(32, 32, 3), num_classes=10):
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ], name="aug")

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        aug,  # <-- CINIC pretraining augmentation

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   # 32 -> 16

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   # 16 -> 8

        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),

        layers.GlobalAveragePooling2D(),                          # -> (128,)
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ===============================================================================

def build_tinyimagenet_base_v1(input_shape=(64, 64, 3), num_classes=200):

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),  # 64 -> 32 (if input 64)

        # Block 2
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),  # 32 -> 16

        # Block 3
        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=True),

        # Fully-convolutional head:
        # 1x1 conv to class logits map (H x W x num_classes)
        layers.Conv2D(num_classes, kernel_size=1, padding="same", use_bias=True),

        # Global average pool across H,W -> (num_classes,)
        layers.GlobalAveragePooling2D(),

        layers.Activation("softmax"),
    ], name="tinyimagenet_fcconv_v1")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# ===============================================================================

def build_model_by_version(version: str, input_shape, num_classes: int):

    match version:
        case "v1":
            model = build_cifar_base(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v1"

        case "v2":
            model = build_cifar_base_v2(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v2"

        case "v3":
            model = build_cifar_base_v3(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v3"

        case "v4":
            model = build_cifar_base_v4(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v4"

        case "v5":
            model = build_cifar_base_v5(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v5"

        case "v6":
            model = build_cifar_base_v6(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v6"

        case "v5_cinic":         # NEW: CINIC pretrain version

            model = build_cinic_base_v1(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cinic10_base_plus_head_v1"

        case "v1_cifar100":

            model = build_cifar_base(input_shape=input_shape, num_classes=100)
            name_h5_file = "cifar100_pretrained_base"

        case "v1_tinyimagenet":
            # Pretrain on Tiny ImageNet-200
            model = build_tinyimagenet_base_v1(input_shape=input_shape, num_classes=200)
            name_h5_file = "tinyimagenet200_pretrained_v2"

        case _:
            raise ValueError(f"Unknown version: {version}")

    return model, name_h5_file

# ===============================================================================

def train_and_export(out_dir="pretrained_model", batch_size=128):
    
    # version = "v2"
    version = "v6"
    # version = "v5_cinic"
    # version = "v1_cifar100"
    # version = "v1_tinyimagenet"

    EPOCHS = 10

    if "cinic" in version: # ================================================================================

            train_ds, val_ds, test_ds = load_cinic10("../data/DS_10283_3192/", batch_size=128)
            model, name_h5_file = build_model_by_version(version, (32,32,3), 10)
            history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    elif "tinyimagenet" in version:     # ================================================================================

        tiny_root = "../data/tiny-imagenet/tiny-imagenet-200"
        train_ds, val_ds = load_tiny_imagenet200(tiny_root, batch_size=batch_size, img_size=(64, 64))

        model, name_h5_file = build_model_by_version(version, (64, 64, 3), 200)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        ]

        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    else:   # ============================================================================================================

        if "cifar100" in version: 
            x_train, y_train, x_test, y_test = load_cifar100()
        else:
            x_train, y_train, x_test, y_test = load_cifar10()

        model, name_h5_file = build_model_by_version(version, (32,32,3), 10)
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
        ]

        history = model.fit(
            x_train, y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=batch_size,
            verbose=2,
            callbacks=callbacks
        )
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"\nCIFAR-10 test acc: {test_acc:.4f}, loss: {test_loss:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, f"{name_h5_file}.h5")
    log_path = os.path.join(out_dir, f"{name_h5_file}.txt")

    model.save(h5_path)
    print("Saved Keras H5:", h5_path)

    return model, history

# ===============================================================================

if __name__ == "__main__":
    
    # export_mobilenetv2_base()
    # export_mobilenetv2_base_224()
    # export_mobilenet_base_224()

    # export_mobilenetv3small_base()
    train_and_export()


    # source ~/venvs/tf215/bin/activate

# Ranking of overfiiting highest to lowest:
# build_cifar_base_v3 (Flatten + Dense)
# build_cifar_base (v1) (heavier conv stack + 128 features head)
# build_cifar_base_v4 (heavier conv stack but GAP helps)
# build_cifar_base_v2 (smallest; GAP; lowest capacity
# MobileNetV2 / MobileNetV3Small (frozen base) => this provides with the least overfitting
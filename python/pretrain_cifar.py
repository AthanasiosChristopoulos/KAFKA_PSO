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
import tensorflow_datasets as tfds


DATASET = "cifar10"
CINIC_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]
@contextmanager
def tee_output(log_path: str):
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

    with open(log_path, "w", buffering=1) as f:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = Tee(old_out, f)
        sys.stderr = Tee(old_err, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

# ================================================================================================

def export_mobilenetv2_base(save_path="pretrained_model/mobilenetv2_base_32x32.h5"):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,     
        weights="imagenet"
    )

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

def load_cifar10(half):

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    y_train = y_train.astype("int64").reshape(-1)
    y_test  = y_test.astype("int64").reshape(-1)

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    if(half):
        rng = np.random.default_rng(123)

        idx = rng.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]

        half_idx = len(x_train) // 2

        x_train = x_train[half_idx:]
        y_train = y_train[half_idx:]
        
    return x_train, y_train, x_test, y_test

def load_cifar10(half=True):

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    y_train = y_train.astype("int64").reshape(-1)
    y_test  = y_test.astype("int64").reshape(-1)

    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # -------------------------------------------------
    # deterministic split
    # -------------------------------------------------
    rng = np.random.default_rng(123)

    idx = rng.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    half_idx = len(x_train) // 2

    x_train = x_train[half_idx:]
    y_train = y_train[half_idx:]

    return x_train, y_train, x_test, y_test

# ===============================================================================

# def load_cifar5(classes=(0, 1, 2, 3, 4)):
def load_cifar5(classes=(0, 1, 4, 8, 9)): # airplane, automobile, deer, ship, truck
    
    print(f"Loading classes {classes}")
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    y_train = y_train.astype("int64").reshape(-1)
    y_test  = y_test.astype("int64").reshape(-1)

    classes = np.array(classes, dtype=np.int64)

    train_mask = np.isin(y_train, classes)
    test_mask  = np.isin(y_test, classes)

    x_train = x_train[train_mask].astype("float32") / 255.0
    y_train = y_train[train_mask]

    x_test = x_test[test_mask].astype("float32") / 255.0
    y_test = y_test[test_mask]

    remap = {int(c): i for i, c in enumerate(classes.tolist())}
    y_train = np.vectorize(remap.get)(y_train).astype(np.int64)
    y_test  = np.vectorize(remap.get)(y_test).astype(np.int64)

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
    if out_dir.exists() and any(out_dir.iterdir()):
        return str(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    with ann_path.open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                img_name, wnid = parts[0], parts[1]
                mapping[img_name] = wnid

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

    norm = layers.Rescaling(1.0 / 255.0)

    def _norm_map(x, y):
        return norm(x), y

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(_norm_map, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.map(_norm_map, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds

# ===============================================================================
# Model (Simple CIFAR feature extractor + head)

def build_cifar_base(input_shape=(32, 32, 3), num_classes=10):   
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),  

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),   

        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),

        layers.GlobalAveragePooling2D(),              
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
        layers.MaxPooling2D(2), 

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2), 

        layers.Conv2D(feat_dim, 3, padding="same", activation="relu", use_bias=True),

        layers.GlobalAveragePooling2D(),  
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
        layers.MaxPooling2D(2), 

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2), 

        layers.Conv2D(feat_dim, 3, padding="same", activation="relu", use_bias=True),

        layers.GlobalAveragePooling2D(),      
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

        layers.Flatten(),   
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
# Epoch 20/20
# 352/352 - 3s - loss: 0.2835 - accuracy: 0.9038 - val_loss: 0.5756 - val_accuracy: 0.8236 - lr: 2.5000e-04 - 3s/epoch - 9ms/step
def build_cifar_base_v4(input_shape=(32, 32, 3), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 32 -> 16

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 16 -> 8

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

        layers.Conv2D(32, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 32 -> 16

        layers.Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 16 -> 8

        layers.Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True),

        layers.Flatten(),

        layers.Dense(64, activation="relu", use_bias=True),
        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ===============================================================================
# Epoch 19/20
# 352/352 - 4s - loss: 0.0053 - accuracy: 0.9998 - val_loss: 1.5402 - val_accuracy: 0.7860 - lr: 1.5625e-05 - 4s/epoch - 10ms/step

def build_cifar_base_v5(input_shape=(32, 32, 3), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 32 -> 16

        layers.Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 16 -> 8

        layers.Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", use_bias=True),

        layers.Flatten(),

        layers.Dense(64, activation="relu", use_bias=True),
        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ===============================================================================
# Epoch 20/20
# 352/352 - 5s - loss: 0.2228 - accuracy: 0.9230 - val_loss: 0.4664 - val_accuracy: 0.8574 - lr: 3.1250e-05 - 5s/epoch - 13ms/step
def build_cifar_base_v5_1(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 32 -> 16
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 16 -> 8
        layers.Dropout(0.30),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Flatten(),

        layers.Dense(64, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.40),

        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ===============================================================================
# Epoch 20/20
# 352/352 - 4s - loss: 0.6116 - accuracy: 0.8236 - val_loss: 0.7065 - val_accuracy: 0.8088 - lr: 2.5000e-04 - 4s/epoch - 10ms/step
def build_cifar_base_v5_2(input_shape=(32, 32, 3), num_classes=10):
    wd = 1e-4

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(
            32, (3, 3), padding="same", activation="relu", use_bias=True,
            kernel_regularizer=keras.regularizers.l2(wd)
        ),
        layers.Conv2D(
            32, (3, 3), padding="same", activation="relu", use_bias=True,
            kernel_regularizer=keras.regularizers.l2(wd)
        ),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 32 -> 16
        layers.Dropout(0.20),

        layers.Conv2D(
            64, (3, 3), padding="same", activation="relu", use_bias=True,
            kernel_regularizer=keras.regularizers.l2(wd)
        ),
        layers.Conv2D(
            64, (3, 3), padding="same", activation="relu", use_bias=True,
            kernel_regularizer=keras.regularizers.l2(wd)
        ),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 16 -> 8
        layers.Dropout(0.30),

        layers.Conv2D(
            128, (3, 3), padding="same", activation="relu", use_bias=True,
            kernel_regularizer=keras.regularizers.l2(wd)
        ),
        layers.Conv2D(
            128, (3, 3), padding="same", activation="relu", use_bias=True,
            kernel_regularizer=keras.regularizers.l2(wd)
        ),

        layers.Flatten(),

        layers.Dense(
            64, activation="relu", use_bias=True,
            kernel_regularizer=keras.regularizers.l2(wd)
        ),
        layers.Dropout(0.40),

        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ===============================================================================

# Epoch 20/20
# 352/352 - 3s - loss: 0.4395 - accuracy: 0.8434 - val_loss: 0.5566 - val_accuracy: 0.8186 - lr: 2.5000e-04 - 3s/epoch - 9ms/step
def build_cifar_base_v5_3(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.Activation("relu"),

        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 32 -> 16
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.Activation("relu"),

        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 16 -> 8
        layers.Dropout(0.30),

        layers.Flatten(),

        layers.Dense(64, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.40),

        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ===============================================================================
# Epoch 18/20
# 352/352 - 3s - loss: 0.0043 - accuracy: 1.0000 - val_loss: 1.5378 - val_accuracy: 0.7862 - lr: 6.2500e-05 - 3s/epoch - 10ms/step
# Overfitting in the orphan shack

def build_cifar_base_v6(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 32 -> 16

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 16 -> 8

        layers.Conv2D(96, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(96, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 8 -> 4

        layers.Conv2D(96, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),  # 4 -> 2

        layers.Flatten(),

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

        aug, 
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

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),  

        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),  

        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(256, 3, padding="same", activation="relu", use_bias=True),

        layers.Conv2D(num_classes, kernel_size=1, padding="same", use_bias=True),

        layers.GlobalAveragePooling2D(),

        layers.Activation("softmax"),
    ], name="tinyimagenet_fcconv_v1")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# =============================================================================
# STL-10 (96x96) -> downsample to 32x32

def load_stl10_32(
    data_dir="./data",
    batch_size=128,
    val_split=0.1,
    shuffle_buffer=10_000,
    seed=42,
    return_tfdata=True,
):

    os.makedirs(data_dir, exist_ok=True)

    ds_train = tfds.load("stl10", split="train", data_dir=data_dir, as_supervised=True)
    ds_test  = tfds.load("stl10", split="test",  data_dir=data_dir, as_supervised=True)

    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess(x, y):
        x = tf.image.resize(x, (32, 32), method="bilinear", antialias=True)
        x = tf.cast(x, tf.float32) / 255.0
        y = tf.cast(y, tf.int32)
        return x, y

    ds_train = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds_test  = ds_test.map(preprocess,  num_parallel_calls=AUTOTUNE)

    train_count = tf.data.experimental.cardinality(ds_train).numpy()
    if train_count < 0:
        train_count = 5000

    val_count = int(train_count * val_split)
    train_count2 = train_count - val_count

    ds_train = ds_train.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds_val = ds_train.take(val_count)
    ds_train = ds_train.skip(val_count)

    ds_train = ds_train.batch(batch_size).prefetch(AUTOTUNE)
    ds_val   = ds_val.batch(batch_size).prefetch(AUTOTUNE)
    ds_test  = ds_test.batch(batch_size).prefetch(AUTOTUNE)

    if return_tfdata:
        return ds_train, ds_val, ds_test

    x_train, y_train = tfds.as_numpy(tfds.dataset_as_numpy(ds_train.unbatch()))

    def ds_to_numpy(ds):
        xs, ys = [], []
        for xb, yb in ds:
            xs.append(xb.numpy())
            ys.append(yb.numpy())
        return tf.concat(xs, axis=0).numpy(), tf.concat(ys, axis=0).numpy()

    ds_train_u = tfds.load("stl10", split=f"train[{val_count}:]", data_dir=data_dir, as_supervised=True).map(preprocess)
    ds_val_u   = tfds.load("stl10", split=f"train[:{val_count}]", data_dir=data_dir, as_supervised=True).map(preprocess)
    ds_test_u  = tfds.load("stl10", split="test", data_dir=data_dir, as_supervised=True).map(preprocess)

    x_train, y_train = ds_to_numpy(ds_train_u.batch(batch_size))
    x_val, y_val     = ds_to_numpy(ds_val_u.batch(batch_size))
    x_test, y_test   = ds_to_numpy(ds_test_u.batch(batch_size))

    return x_train, y_train, x_val, y_val, x_test, y_test


# =============================================================================
# CIFAR-style base model for STL-10

def build_stl10_base_v1(input_shape=(32, 32, 3), num_classes=10):

    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # 32x32
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2), 

        # 16x16
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2), 

        # 8x8
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(128, 3, padding="same", activation="relu", use_bias=True),
        layers.GlobalAveragePooling2D(), 

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

def pretrain_stl10_and_export(
    data_dir="./data",
    out_dir="pretrained_model",
    epochs=20,
    batch_size=128,
):
    train_ds, val_ds, test_ds = load_stl10_32(
        data_dir=data_dir,
        batch_size=batch_size,
        return_tfdata=True
    )

    model = build_stl10_base_v1(input_shape=(32, 32, 3), num_classes=10)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nSTL-10 (downsampled 32x32) test acc: {test_acc:.4f}, loss: {test_loss:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, "stl10_pretrained_base_plus_head_v1.h5")
    model.save(h5_path)
    print("Saved Keras H5:", h5_path)

    return model, history

# ===============================================================================

def build_stl10_resnet20_v1(input_shape=(32, 32, 3), num_classes=10):
    
    def conv3x3(x, filters, stride=1):
        return layers.Conv2D(
            filters, 3, strides=stride, padding="same",
            use_bias=False, kernel_initializer="he_normal"
        )(x)

    def bn_relu(x):
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    def basic_block(x, filters, stride=1):
        shortcut = x

        x = conv3x3(x, filters, stride=stride)
        x = bn_relu(x)
        x = conv3x3(x, filters, stride=1)
        x = layers.BatchNormalization()(x)

        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(
                filters, 1, strides=stride, padding="same",
                use_bias=False, kernel_initializer="he_normal"
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
    x = bn_relu(x)
    
    for _ in range(3):
        x = basic_block(x, 16, stride=1)

    x = basic_block(x, 32, stride=2)
    for _ in range(2):
        x = basic_block(x, 32, stride=1)

    x = basic_block(x, 64, stride=2)
    for _ in range(2):
        x = basic_block(x, 64, stride=1)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", use_bias=True)(x)

    model = keras.Model(inputs, outputs, name="stl10_resnet20_v1")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ===============================================================================

def pretrain_stl10_resnet20_and_export(
    data_dir="./data",
    out_dir="pretrained_model",
    epochs=20,
    batch_size=128,
):
    train_ds, val_ds, test_ds = load_stl10_32(
        data_dir=data_dir,
        batch_size=batch_size,
        return_tfdata=True
    )

    model = build_stl10_resnet20_v1(input_shape=(32, 32, 3), num_classes=10)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nSTL-10 (downsampled 32x32) test acc: {test_acc:.4f}, loss: {test_loss:.4f}")

    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, "stl10_pretrained_resnet20_v1.h5")
    model.save(h5_path)
    print("Saved Keras H5:", h5_path)

    return model, history

# ===============================================================================

def build_model_by_version(version: str, input_shape, num_classes: int, cifar_5_classes):

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
            
        case "v5_1":
            model = build_cifar_base_v5_1(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v5_1"
            
        case "v5_2":
            model = build_cifar_base_v5_2(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v5_2"
 
        case "v5_3":
            model = build_cifar_base_v5_3(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v5_3"
            
        case "v4_cifar5":
            model = build_cifar_base_v4(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = f"cifar5_base_plus_head_v4_{cifar_5_classes}"
            
        case "v5_cifar5":
            model = build_cifar_base_v5(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = f"cifar5_base_plus_head_v5_{cifar_5_classes}"

        # Epoch 20/20
        # 176/176 - 2s - loss: 0.1005 - accuracy: 0.9660 - val_loss: 0.2971 - val_accuracy: 0.9108 - lr: 2.5000e-04 - 2s/epoch - 9ms/step  
        case "v5_1_cifar5":
            model = build_cifar_base_v5_1(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = f"cifar5_base_plus_head_v5_1_{cifar_5_classes}"
                        
        # Epoch 20/20
        # 176/176 - 2s - loss: 0.3430 - accuracy: 0.9048 - val_loss: 0.3606 - val_accuracy: 0.9028 - lr: 0.0010 - 2s/epoch - 10ms/step
        case "v5_2_cifar5":
            model = build_cifar_base_v5_2(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = f"cifar5_base_plus_head_v5_2_{cifar_5_classes}"

        case "v6_cifar5":
            model = build_cifar_base_v6(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = f"cifar5_base_plus_head_v6_{cifar_5_classes}"
                              
        case "v5_cifar100":
            model = build_cifar_base_v5(input_shape=input_shape, num_classes=100)
            name_h5_file = "cifar100_base_plus_head_v5"

        case "v6":
            model = build_cifar_base_v6(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v6"

        case "v5_cinic": 

            model = build_cinic_base_v1(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cinic10_base_plus_head_v1"

        case "v1_cifar100":

            model = build_cifar_base(input_shape=input_shape, num_classes=100)
            name_h5_file = "cifar100_pretrained_base"

        case "v1_tinyimagenet":
            model = build_tinyimagenet_base_v1(input_shape=input_shape, num_classes=200)
            name_h5_file = "tinyimagenet200_pretrained_v2"

        case _:
            raise ValueError(f"Unknown version: {version}")

    return model, name_h5_file

# ===============================================================================

def train_and_export(out_dir="pretrained_model", batch_size=128):
    
    # version = "v2"
    version = "v5_3"
    # version = "v5_cinic"
    # version = "v5_cifar100"
    # version = "v4_cifar5"
    # version = "v5_cifar5"
    # version = "v5_1_cifar5"
    # version = "v5_2_cifar5"
    # version = "v1_tinyimagenet"
    # version = "v1_stl10"
    # version = "v1_stl10_resnet20"
    
    classes_for_cifar5 = (0, 1, 4, 8, 9)
    # classes_for_cifar5 = (5, 6, 7, 8, 9)

    EPOCHS = 20
    
    half = True

    if "cinic" in version: # ================================================================================

        train_ds, val_ds, test_ds = load_cinic10("../data/DS_10283_3192/", batch_size=128)
        model, name_h5_file = build_model_by_version(version, (32,32,3), 10)
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
            
    elif "v1_stl10_resnet20" in version:
        pretrain_stl10_resnet20_and_export(
            data_dir="./data",
            out_dir="pretrained_model",
            epochs=EPOCHS,
            batch_size=batch_size,
        )
        exit(0)

    elif "v1_stl10" in version:     # ================================================================================
        
        pretrain_stl10_and_export(
            data_dir="./data",       
            out_dir="pretrained_model",
            epochs=20,
            batch_size=128,
        )

        exit(0)

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
            num_classes = 100
            
        elif "cifar5" in version:
            x_train, y_train, x_test, y_test = load_cifar5(classes_for_cifar5)
            num_classes = 5
        
        else:
            
            x_train, y_train, x_test, y_test = load_cifar10(half)
            num_classes = 10

        model, name_h5_file = build_model_by_version(version, (32,32,3), num_classes, str(classes_for_cifar5))
        
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
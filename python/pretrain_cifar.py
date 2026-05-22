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

# ============================================================================================

def evaluate_dataset(X_train, y_train, X_test, y_test, n_classes=7):
    
    if(X_train is not None and y_train is not None):
        print(
            "Train shape:", X_train.shape,
            "classes / y (labels):", (int(y_train.min()), int(y_train.max())),
            "with counts:", np.bincount(y_train, minlength=n_classes)
        )
        
    if(X_test is not None and y_test is not None):
        print(
            "Test shape:", X_test.shape,
            "classes / y (labels):", (int(y_test.min()), int(y_test.max())),
            "with counts:", np.bincount(y_test, minlength=n_classes)
        )

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

    if half:
        rng = np.random.default_rng(123)   

        idx = rng.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]

        half_idx = len(x_train) // 2
        x_train = x_train[half_idx:]
        y_train = y_train[half_idx:]

        idx_half = rng.permutation(len(x_train))
        x_train = x_train[idx_half]
        y_train = y_train[idx_half]

    return x_train, y_train, x_test, y_test

# ===============================================================================

def load_cifar5(classes=(0, 1, 4, 8, 9)):   # airplane, automobile, deer, ship, truck
    
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
# Model

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

        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),
        layers.Dropout(0.30),

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

        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),

        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.Activation("relu"),

        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),  # 32 -> 16
        layers.Dropout(0.25),

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
        layers.MaxPooling2D(2), 

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2), 

        layers.Conv2D(96, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(96, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2),

        layers.Conv2D(96, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2), 

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

def build_cifar_base_v7(input_shape=(32, 32, 3), num_classes=5):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2), 

        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2), 

        layers.Conv2D(96, 3, padding="same", activation="relu", use_bias=True),
        layers.Conv2D(96, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2), 

        layers.Conv2D(48, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(2), 

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

# ===============================================================================

def build_model_by_version(version: str, input_shape, num_classes: int, cifar_5_classes):

    match version:
        
        # ====================================================================================
        # pretrain on cifar10:
        
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
            
        case "v6":
            model = build_cifar_base_v6(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v6"
            
        case "v7":
            model = build_cifar_base_v7(input_shape=input_shape, num_classes=num_classes)
            name_h5_file = "cifar10_base_plus_head_v7"
            
        # ====================================================================================
        # pretrain on cifar5:

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
        
        # ====================================================================================
        # pretrain on cifar100:

        case "v1_cifar100":

            model = build_cifar_base(input_shape=input_shape, num_classes=100)
            name_h5_file = "cifar100_pretrained_base"
            
        case "v4_cifar100":
            # 352/352 - 3s - loss: 1.2441 - accuracy: 0.6519 - val_loss: 2.1159 - val_accuracy: 0.4664 - lr: 2.5000e-04 - 3s/epoch - 8ms/step
            model = build_cifar_base_v4(input_shape=input_shape, num_classes=100)
            name_h5_file = "cifar100_pretrained_base_v4"
              
        case "v5_cifar100":
            model = build_cifar_base_v5(input_shape=input_shape, num_classes=100)
            name_h5_file = "cifar100_base_plus_head_v5"
        case _:
            raise ValueError(f"Unknown version: {version}")

    return model, name_h5_file

# ===============================================================================

def train_and_export(out_dir="pretrained_model", batch_size=128):
    
    # version = "v2"
    # version = "v5"
    # version = "v6"
    # version = "v7"
    version = "v4_cifar100"
    # version = "v5_cifar100"
    # version = "v4_cifar5"
    # version = "v5_cifar5"
    # version = "v5_1_cifar5"
    # version = "v5_2_cifar5"

    # classes_for_cifar5 = (0, 1, 4, 8, 9)
    # classes_for_cifar5 = (5, 6, 7, 8, 9)

    classes_for_cifar5 = (-2)
    print(classes_for_cifar5)
    
    EPOCHS = 20
    
    half = False
    
    # ======================================================================
    if "cifar100" in version: 
        x_train, y_train, x_test, y_test = load_cifar100()
        evaluate_dataset(x_train, y_train, x_test, y_test, 100)
        num_classes = 100
        
    # ======================================================================
    elif "cifar5" in version:
        x_train, y_train, x_test, y_test = load_cifar5(classes_for_cifar5)
        num_classes = 5
    
    # ======================================================================
    else:   # in the case of pretraining with cifar10
        
        x_train, y_train, x_test, y_test = load_cifar10(half)
        num_classes = 10

    model, name_h5_file = build_model_by_version(version, (32, 32, 3), num_classes, str(classes_for_cifar5))
    
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
    
    if half == True:
        h5_path = os.path.join(out_dir, f"{name_h5_file}_half.h5")
    else:
        h5_path = os.path.join(out_dir, f"{name_h5_file}.h5")
        
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
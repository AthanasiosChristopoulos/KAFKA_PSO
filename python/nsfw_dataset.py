import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# Config
# ============================================================

DATA_DIR = "../data/nsfw_dataset_v1"    

IMAGE_SIZE = 32          # set to 64 or 32
# IMAGE_SIZE = 64          # set to 64 or 32

BATCH_SIZE = 32
SEED = 123
EPOCHS = 10
VAL_SPLIT = 0.2

IMG_HEIGHT = IMAGE_SIZE
IMG_WIDTH = IMAGE_SIZE
AUTOTUNE = tf.data.AUTOTUNE

# ============================================================
# Load datasets
# ============================================================

print("Current working directory:", os.getcwd())
print("DATA_DIR:", DATA_DIR)
print("Absolute DATA_DIR:", os.path.abspath(DATA_DIR))
print("Exists:", os.path.exists(DATA_DIR))
print("Contents:", os.listdir(DATA_DIR)[:10])

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),   # downsampling happens here
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="int",
)

# Save class info BEFORE map()
class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)
print("Image size:", IMAGE_SIZE, "x", IMAGE_SIZE)

expected = {"drawings", "hentai", "neutral", "porn", "sexy"}
if set(class_names) != expected:
    print("\n[WARNING] Folder names do not match the standard 5-class NSFW set.")
    print("Expected:", sorted(expected))
    print("Found   :", class_names)
    print()
    exit(0)

# ============================================================
# Preprocessing
# ============================================================

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

# ============================================================
# Class weights
# ============================================================

all_labels = []
for _, labels in train_ds.unbatch():
    all_labels.append(int(labels.numpy()))

counts = [0] * num_classes
for y in all_labels:
    counts[y] += 1

total = sum(counts)
class_weight = {}
for i, c in enumerate(counts):
    if c > 0:
        class_weight[i] = total / (num_classes * c)

print("Train counts:", dict(zip(class_names, counts)))

all_labels = []
for _, labels in val_ds.unbatch():
    all_labels.append(int(labels.numpy()))

counts = [0] * num_classes
for y in all_labels:
    counts[y] += 1

total = sum(counts)
class_weight = {}
for i, c in enumerate(counts):
    if c > 0:
        class_weight[i] = total / (num_classes * c)

print("Train counts:", dict(zip(class_names, counts)))
print("Class weights:", class_weight)

# ============================================================
# Performance pipeline
# ============================================================

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ============================================================
# Model
# ============================================================

# def build_custom_cnn(input_shape=(64, 64, 3), num_classes=5, image_size=64):
#     inputs = keras.Input(shape=input_shape)

#     x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    
#     if image_size == 64:
#         x = layers.MaxPooling2D(pool_size=(2, 2))(x)  
        
#     x = layers.Flatten()(x)
#     outputs = layers.Dense(num_classes, activation="softmax")(x)

#     return keras.Model(inputs, outputs)

# ============================================================

# def build_custom_cnn(input_shape=(32, 32, 3), num_classes=5, image_size=64):
#     inputs = keras.Input(shape=input_shape)

#     x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)   # 32 -> 16

#     x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)   # 16 -> 8

#     x = layers.Conv2D(96, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)
#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)   # 8 -> 4

#     x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)                            # 4 x 4 x 128

#     x = layers.GlobalAveragePooling2D()(x)         # -> 128

#     # optional regularization only for GD pretraining
#     x = layers.Dropout(0.25)(x)

#     outputs = layers.Dense(num_classes, activation="softmax")(x)

#     return keras.Model(inputs, outputs, name="nsfw_cnn_gap_32_bn")

# ============================================================

# def build_custom_cnn(input_shape=(64, 64, 3), num_classes=5, image_size=64):
#     inputs = keras.Input(shape=input_shape)

#     # Block 1: 64 -> 32
#     x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     # Block 2: 32 -> 16
#     x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     # Block 3: 16 -> 8
#     x = layers.Conv2D(96, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.Conv2D(96, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     # Block 4: 8 -> 8
#     x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     # Compact PSO-friendly representation
#     x = layers.GlobalAveragePooling2D()(x)   # -> 128
#     x = layers.Dropout(0.25)(x)

#     outputs = layers.Dense(num_classes, activation="softmax")(x)

#     return keras.Model(inputs, outputs, name="nsfw_cnn_gap_64_bn")

# ============================================================

# def build_custom_cnn(input_shape=(32, 32, 3), num_classes=5,  image_size=32):
    
#     inputs = keras.Input(shape=input_shape)

#     # Block 1: 32 -> 16
#     x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     # Block 2: 16 -> 8
#     x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)

#     # Block 3: 8 -> 4
#     x = layers.Conv2D(96, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.ReLU()(x)

#     x = layers.MaxPooling2D(pool_size=(2, 2))(x)   # -> 4x4x128

#     x = layers.Flatten()(x)                        # 2048

#     # Dense layer 1: keep this small for PSO-friendly transfer
#     x = layers.Dense(64, activation="relu")(x)

#     # GD-only regularization
#     x = layers.Dropout(0.30)(x)

#     # Dense layer 2: final classifier
#     outputs = layers.Dense(num_classes, activation="softmax")(x)

#     return keras.Model(inputs, outputs, name="nsfw_cnn_flat_32_two_dense")

def build_custom_cnn(input_shape=(32, 32, 3), num_classes=5, image_size=32):
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

        # Optional extra pooling only for 64x64 version
        # For 32x32 this stays at 8x8x128
        # For 64x64 it becomes 8x8 after one extra pool if desired
        # but for your current 32x32 setup this branch does nothing
        # and keeps the representation larger.
        # if image_size == 64:
        #     layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"),

        layers.Flatten(),

        layers.Dense(64, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.40),

        layers.Dense(num_classes, activation="softmax", use_bias=True),
    ])

    return model

model = build_custom_cnn(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    num_classes=num_classes,
    image_size=IMAGE_SIZE,
)
model.summary()

# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
    ],
)

# ============================================================
# Output filenames
# ============================================================

final_h5_path = f"pretrained_model/nsfw_base_plus_head_{IMAGE_SIZE}_v1.h5"

# ============================================================
# Callbacks
# ============================================================

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    ),
    keras.callbacks.ModelCheckpoint(
        final_h5_path,
        monitor="val_accuracy",
        save_best_only=True,
    ),
]

# ============================================================
# Train
# ============================================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks,
)

# ============================================================
# Final evaluation
# ============================================================

results = model.evaluate(val_ds, verbose=1)
print("\nValidation results:")
for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# ============================================================
# Save model
# ============================================================

model.save(final_h5_path)

print(f"Saved H5 model to   : {final_h5_path}")


# image_dataset_from_directory() explained: 
# This function assumes a directory structure like this. Each folder name becomes a class label:
# This is expecting this 2 tiered file structure

# DATA_DIR/
#     class1/
#         img1.jpg
#         img2.jpg
#         img3.jpg
#     class2/
#         img4.jpg
#         img5.jpg
#     class3/
#         ...

# So TensorFlow internally does something like:
# drawings -> label 0
# hentai   -> label 1
# neutral  -> label 2
# porn     -> label 3
# sexy     -> label 4
# Then it scans all files.

# find nsfw_dataset_v1 -type f | wc -l
# find sexy -type f | wc -l
# find neutral -type f | wc -l
# find drawings -type f | wc -l
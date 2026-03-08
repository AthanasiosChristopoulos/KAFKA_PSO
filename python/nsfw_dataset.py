import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# Config
# ============================================================
DATA_DIR = "../data"
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
SEED = 123
EPOCHS = 20
VAL_SPLIT = 0.2

AUTOTUNE = tf.data.AUTOTUNE

# Expected folder structure:
# /pwd/data/
#   drawings/
#   hentai/
#   neutral/
#   porn/
#   sexy/

# ============================================================
# Load datasets
# ============================================================
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

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

# Safety check
expected = {"drawings", "hentai", "neutral", "porn", "sexy"}
if set(class_names) != expected:
    print("\n[WARNING] Your folder names are not the standard 5-class NSFW ones.")
    print("Expected:", sorted(expected))
    print("Found   :", class_names)
    print("Training will still run, but confirm the dataset is the one you mean.\n")

# ============================================================
# Optional: class weights for imbalance
# ============================================================
# This dataset family is often imbalanced in practice, so class weights help.
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
print("Class weights:", class_weight)

# ============================================================
# Performance pipeline
# ============================================================
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ============================================================
# Data augmentation
# Keep it mild because aggressive transforms may distort semantics
# ============================================================
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="augmentation",
)

# ============================================================
# Custom lightweight CNN
# Much smaller than MobileNet / EfficientNet
# ============================================================
def build_custom_cnn(input_shape=(64, 64, 3), num_classes=5):
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)

    # small dense head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="nsfw_custom_cnn_64")

model = build_custom_cnn(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    num_classes=num_classes,
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
        "best_nsfw_custom_cnn_64.keras",
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
# Save final model
# ============================================================
model.save("final_nsfw_custom_cnn_64.keras")
print("\nSaved model to final_nsfw_custom_cnn_64.keras")
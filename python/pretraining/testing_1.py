#!/usr/bin/env python3
"""
CIFAR-10 Transfer Learning with DenseNet121 (ImageNet pretrained)

What this script does:
1) Loads CIFAR-10 (32x32 RGB images)
2) Preprocesses inputs using DenseNet preprocess_input
3) One-hot encodes labels
4) Resizes images from 32x32 to 224x224
5) Builds a model:
      Input(32x32x3)
        -> Resize(224x224x3)
        -> DenseNet121 (frozen)
        -> Flatten
        -> Dense(500, relu)
        -> Dropout(0.2)
        -> Dense(10, softmax)
6) Trains for a few epochs
7) Saves to cifar10.h5
8) Reloads the model and evaluates (optional)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # reduce TF logs (must be before importing TF)

import tensorflow as tf


# ---------------------------
# 1) Preprocessing helpers
# ---------------------------

def preprocess_cifar10(X, Y):
    """
    X: np array (N, 32, 32, 3), uint8 in [0..255]
    Y: np array (N, 1) or (N,), integer labels 0..9

    Returns:
      X_p: float32 preprocessed for DenseNet
      Y_p: one-hot labels (N, 10)
    """
    # Ensure correct shapes
    Y = Y.reshape(-1)  # (N,)

    # DenseNet preprocess expects float images
    X = X.astype("float32")

    # Apply DenseNet preprocessing (scales channels as ImageNet expects)
    X_p = tf.keras.applications.densenet.preprocess_input(X)

    # One-hot encode labels
    Y_p = tf.keras.utils.to_categorical(Y, num_classes=10)

    return X_p, Y_p


# ---------------------------
# 2) Model builder
# ---------------------------

def build_model():

    inputs = tf.keras.Input(shape=(32, 32, 3), name="cifar10_input")

    x = tf.keras.layers.Resizing(224, 224, name="resize_224")(inputs)   # Resizing for 224

    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )

    # Freeze base weights (important to do before compile)
    base.trainable = False

    # Entire model:
    x = base(x, training=False)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(500, activation="relu", name="dense_500")(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_02")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifar10_densenet121_transfer")
    return model


# ---------------------------
# 3) Train + Save
# ---------------------------

def train_and_save(
    out_path="cifar10.h5",
    batch_size=128,
    epochs=5
):
    # Load CIFAR-10
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

    # Preprocess data
    X_train_p, Y_train_p = preprocess_cifar10(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_cifar10(X_test, Y_test)

    # Build model
    model = build_model()
    model.summary()

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks (optional but helpful)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]

    # Train
    model.fit(
        X_train_p, Y_train_p,
        validation_data=(X_test_p, Y_test_p),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks
    )

    # Save as H5 (legacy format; good for older pipelines)
    model.save(out_path)
    print(f"\nSaved model to: {out_path}")

    return out_path


# ---------------------------
# 4) Reload + Evaluate
# ---------------------------

def reload_and_evaluate(model_path="cifar10.h5", batch_size=128):
    # Load CIFAR-10 test set
    (_, _), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
    X_test_p, Y_test_p = preprocess_cifar10(X_test, Y_test)

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Evaluate
    loss, acc = model.evaluate(X_test_p, Y_test_p, batch_size=batch_size, verbose=2)
    print(f"\nReloaded model accuracy: {acc:.4f}, loss: {loss:.4f}")


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # NOTE:
    # batch_size=300 may be too big for laptop GPUs (RTX 3050).
    # If you get OOM, use 128 or 64.
    saved_path = train_and_save(out_path="cifar10.h5", batch_size=128, epochs=5)
    reload_and_evaluate(saved_path, batch_size=128)
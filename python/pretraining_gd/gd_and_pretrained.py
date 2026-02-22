import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
import tensorflow as tf
from tensorflow.keras import layers, models

# DATASET="MNIST"
DATASET="CIFAR"

if (DATASET == "CIFAR"):
    # ------------------------------------------------------------
    # 1) Load CIFAR-10
    # ------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # MobileNetV2 expects float + its preprocessing
    x_train = tf.cast(x_train, tf.float32)
    x_test  = tf.cast(x_test,  tf.float32)

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    x_train = preprocess(x_train)
    x_test  = preprocess(x_test)

    # ------------------------------------------------------------
    # 2) Build frozen MobileNetV2 base (feature extractor)
    # ------------------------------------------------------------
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    # ------------------------------------------------------------
    # 3) Add pooling + dense head
    # ------------------------------------------------------------
    inputs = layers.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)     # IMPORTANT: keep BN in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)                # optional
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # ------------------------------------------------------------
    # 4) Train only the head
    # ------------------------------------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=5000,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy (frozen base):", test_acc)

# ======================================================================================================

elif(DATASET == "MNIST"):

    # ------------------------------------------------------------
    # 1) Load MNIST
    # ------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # x: (N, 28, 28) uint8 -> float32
    x_train = tf.cast(x_train, tf.float32)
    x_test  = tf.cast(x_test,  tf.float32)

    # Add channel dim: (N, 28, 28, 1)
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test  = tf.expand_dims(x_test,  axis=-1)

    # Resize to 32x32 (so it matches your CIFAR setup)
    x_train = tf.image.resize(x_train, (32, 32))
    x_test  = tf.image.resize(x_test,  (32, 32))

    # Convert grayscale -> RGB by repeating channels: (N, 32, 32, 3)
    x_train = tf.image.grayscale_to_rgb(x_train)
    x_test  = tf.image.grayscale_to_rgb(x_test)

    # MobileNetV2 preprocessing
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    x_train = preprocess(x_train)
    x_test  = preprocess(x_test)

    # ------------------------------------------------------------
    # 2) Build frozen MobileNetV2 base (feature extractor)
    # ------------------------------------------------------------
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    # ------------------------------------------------------------
    # 3) Add pooling + dense head (simple)
    # ------------------------------------------------------------
    inputs = layers.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)  # keep BN in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    # ------------------------------------------------------------
    # 4) Train only the head
    # ------------------------------------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]

    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("MNIST Test accuracy (frozen base):", test_acc)

    # Optional: save for later (H5)
    model.save("pretrained_model/mnist_mobilenetv2_frozen_head.h5")
    print("Saved H5: pretrained_model/mnist_mobilenetv2_frozen_head.h5")
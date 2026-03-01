import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# DATASET="MNIST"
DATASET="CIFAR"

def cifar_model_v1():
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
    base_model.trainable = True

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
        batch_size=128,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("Test accuracy (frozen base):", test_acc)




# ======================================================================================================

def cifar_model_v2():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # ---- One-hot labels (like the other guy)
    y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
    y_test_oh  = tf.keras.utils.to_categorical(y_test, 10)

    # ---- Preprocess images for MobileNet (v1)
    x_train = tf.cast(x_train, tf.float32)
    x_test  = tf.cast(x_test,  tf.float32)

    preprocess = tf.keras.applications.mobilenet.preprocess_input
    x_train = preprocess(x_train)
    x_test  = preprocess(x_test)

    base_model = tf.keras.applications.MobileNet(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = True # Nesseary need to adapt to different input dimensionality

    inputs = layers.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=True)  
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(10, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),  # lower LR since base is training
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
    ]

    history = model.fit(
        x_train, y_train_oh,
        validation_split=0.1,
        epochs=20,
        batch_size=100,          # like his
        callbacks=callbacks,
        verbose=2
    )

    test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)
    print("Test accuracy (MobileNet v1, trainable base):", test_acc)

# ======================================================================================================

def cifar_model_v3():

    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    NUM_CLASSES = 10
    EPOCHS = 10
    SEED = 42

    tf.random.set_seed(SEED)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    def preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, IMG_SIZE, method="bilinear")
        image = preprocess_input(image)  # -> float32 in [-1, 1], basically normalize
        label = tf.one_hot(tf.cast(label[0], tf.int32), NUM_CLASSES)    # 3  → [0,0,0,1,0,0,0,0,0,0] one hot lavel encoders
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(50000, seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)        # Applies preprocessing function to each sample in the tensor
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    for images, labels in train_ds.take(1):
        print("Batch images:", images.shape, images.dtype, "range:", (tf.reduce_min(images).numpy(), tf.reduce_max(images).numpy()))
        print("Batch labels:", labels.shape, labels.dtype)

    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base_model.trainable = False  
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=2,
        batch_size=128
    )

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Loss:     {loss:.4f}")

# ==========================================================================================================
# For MobileNetV3Small uses both reshaping

def cifar_model_v5():

    # -----------------------
    # Config
    # -----------------------
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    NUM_CLASSES = 10
    SEED = 42

    # Two-stage training (transfer learning -> fine-tuning)
    EPOCHS_FROZEN = 5
    EPOCHS_FINETUNE = 10
    FINETUNE_LAST_N_LAYERS = 30

    tf.random.set_seed(SEED)

    # -----------------------
    # Data
    # -----------------------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    def preprocess(image, label):
        # image: uint8 [0,255], label: shape (1,)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, IMG_SIZE, method="bilinear")

        # IMPORTANT: keep labels as integer class ids for sparse loss
        label = tf.squeeze(label, axis=-1)          # (1,) -> ()
        label = tf.cast(label, tf.int32)            # scalar int
        return image, label

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(50000, seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Sanity check (labels should be ints in [0..9])
    for images, labels in train_ds.take(1):
        print("Batch images:", images.shape, images.dtype,
              "range:", (tf.reduce_min(images).numpy(), tf.reduce_max(images).numpy()))
        print("Batch labels:", labels.shape, labels.dtype,
              "range:", (tf.reduce_min(labels).numpy(), tf.reduce_max(labels).numpy()))

    # -----------------------
    # Model: MobileNetV3Small
    # -----------------------
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    # light augmentation (optional but helps CIFAR)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.05)(x)

    base_model = tf.keras.applications.MobileNetV3Small(
        weights="imagenet",
        include_top=False,
        include_preprocessing=True,     # <-- handles the correct preprocessing internally
        input_tensor=x,
        pooling="avg"                  # <-- outputs (None, 576)
    )

    base_model.summary()
    
    base_model.trainable = False

    x = base_model.output
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # -----------------------
    # Stage 1: train head (frozen backbone)
    # -----------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()

    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]

    history_frozen = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks_stage1,
        verbose=2
    )

    # -----------------------
    # Stage 2: fine-tune last N layers
    # -----------------------
    base_model.trainable = True

    # Freeze all but the last N layers of the backbone
    if FINETUNE_LAST_N_LAYERS is not None and FINETUNE_LAST_N_LAYERS > 0:
        for layer in base_model.layers[:-FINETUNE_LAST_N_LAYERS]:
            layer.trainable = False

    # IMPORTANT: lower LR for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    history_finetune = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS_FINETUNE,
        callbacks=callbacks_stage2,
        verbose=2,
        batch_size=128
    )

    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Loss:     {loss:.4f}")



# ======================================================================================================
# ======================================================================================================
# ======================================================================================================
# ======================================================================================================

def mnist_model_v1():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = tf.cast(x_train, tf.float32)
    x_test  = tf.cast(x_test,  tf.float32)
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test  = tf.expand_dims(x_test,  axis=-1)

    x_train = tf.image.resize(x_train, (32, 32))
    x_test  = tf.image.resize(x_test,  (32, 32))

    x_train = tf.image.grayscale_to_rgb(x_train)
    x_test  = tf.image.grayscale_to_rgb(x_test)

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    x_train = preprocess(x_train)
    x_test  = preprocess(x_test)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)  # keep BN in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, outputs)

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

# ================================================================================================

if __name__ == "__main__":
    if(DATASET == "CIFAR"):
        # cifar_model_v1()
        # cifar_model_v2()
        cifar_model_v3()
        # cifar_model_v4()
        # cifar_model_v5()

    else:
        mnist_model_v1()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
import tensorflow as tf
from tensorflow.keras import layers, models

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
    batch_size=128,
    callbacks=callbacks,
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test accuracy (frozen base):", test_acc)

#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BATCH_SIZE = 100
N_WARMUP = 30
N_RUNS = 200

# =========================================================================
def get_cifar_batch():
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x = x_train[:BATCH_SIZE].astype("float32") / 255.0
    return tf.constant(x)

# =========================================================================
def print_env():
    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow:", tf.__version__)
    print("GPU available:", bool(gpus))
    if gpus:
        print("GPU:", gpus[0].name)
    else:
        print("Device: CPU")

# =========================================================================
def build_prebuilt_mobilenetv3small_base():
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        base,
        layers.GlobalAveragePooling2D(),
    ], name="prebuilt_mobilenetv3small_base_gap")
    return model

# =========================================================================
def build_prebuilt_mobilenetv2_base():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        base,
        layers.GlobalAveragePooling2D(),
    ], name="prebuilt_mobilenetv2_base_gap")
    return model


# =========================================================================
def build_prebuilt_mobilenetv1_base():
    base = tf.keras.applications.MobileNet(
        input_shape=(32, 32, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        base,
        layers.GlobalAveragePooling2D(),
    ], name="prebuilt_mobilenetv2_base_gap")
    return model

# =========================================================================
def build_cifar_resnet20_base_gap(input_shape=(32, 32, 3)):
    """
    CIFAR ResNet-20 backbone + GAP.
    Output: feature vector of size 64 (after final block).
    No ImageNet weights. CIFAR-style stem (3x3 conv, no maxpool at start).
    """
    def conv_bn_relu(x, filters, kernel_size=3, stride=1):
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same",
                          use_bias=False, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    def residual_block(x, filters, stride=1):
        shortcut = x

        x = layers.Conv2D(filters, 3, strides=stride, padding="same",
                          use_bias=False, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, strides=1, padding="same",
                          use_bias=False, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)

        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same",
                                     use_bias=False, kernel_initializer="he_normal")(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    inputs = keras.Input(shape=input_shape)

    x = conv_bn_relu(inputs, 16, kernel_size=3, stride=1)
    for _ in range(3):
        x = residual_block(x, 16, stride=1)

    x = residual_block(x, 32, stride=2)

    for _ in range(2):
        x = residual_block(x, 32, stride=1)

    x = residual_block(x, 64, stride=2)
    for _ in range(2):
        x = residual_block(x, 64, stride=1)

    x = layers.GlobalAveragePooling2D()(x)

    model = keras.Model(inputs, x, name="cifar_resnet20_base_gap")
    return model

# =========================================================================
def build_tiny_fcn_gap():
    return keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(),  # 32->16
        layers.Conv2D(32, 3, padding="same", activation="relu", use_bias=True),
        layers.MaxPooling2D(),  # 16->8
        layers.Conv2D(64, 3, padding="same", activation="relu", use_bias=True),
        layers.GlobalAveragePooling2D(),
    ], name="tiny_fcn_gap")

# =========================================================================
def build_simple_flatten_dense():
    return keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(16, 3, padding="valid", activation="relu", use_bias=True),
        layers.Conv2D(32, 3, padding="valid", activation="relu", use_bias=True),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu", use_bias=True),
    ], name="simple_flatten_dense")

# =========================================================================
def benchmark_forward(model, x, use_tf_function=True, jit_compile=False):
    # Build once
    _ = model(x, training=False)

    if use_tf_function:
        @tf.function(jit_compile=jit_compile)
        def fwd(inp):
            return model(inp, training=False)
        forward = fwd
    else:
        forward = lambda inp: model(inp, training=False)

    for _ in range(N_WARMUP):
        _ = forward(x)

    start = time.perf_counter()
    for _ in range(N_RUNS):
        y = forward(x)
    _ = y.numpy()
    end = time.perf_counter()

    return (end - start) / N_RUNS * 1000.0

# ===============================================================

def main(): 
    
    print_env()
    x = get_cifar_batch()

    models = [
        ("Prebuilt MobileNetV3Small base+GAP", build_prebuilt_mobilenetv3small_base()),
        ("Prebuilt MobileNetV2 base+GAP",      build_prebuilt_mobilenetv2_base()),
        ("Prebuilt MobileNetV1 base+GAP",      build_prebuilt_mobilenetv1_base()),
        ("CIFAR ResNet-20 base+GAP",           build_cifar_resnet20_base_gap()),
        ("Tiny fully-conv + GAP",              build_tiny_fcn_gap()),
        ("Simple Flatten+Dense",               build_simple_flatten_dense()),
    ]

    print(f"\nBenchmark: CIFAR batch={BATCH_SIZE}, warmup={N_WARMUP}, runs={N_RUNS}")
    use_tf_function = True
    jit_compile = False
    print(f"use_tf_function={use_tf_function}, jit_compile={jit_compile}\n")

    times = []
    for name, m in models:
        print(f"\n=== {name} ===")
        m.summary()
        t_ms = benchmark_forward(m, x, use_tf_function=use_tf_function, jit_compile=jit_compile)
        times.append((name, t_ms))
        print(f"Avg forward: {t_ms:.3f} ms")

    print("\n\n=== Results (avg ms per forward pass) ===")
    for name, t in sorted(times, key=lambda z: z[1]):
        print(f"{t:8.3f} ms  | {name}")

    fastest = min(t for _, t in times)
    print("\n=== Relative slowdown vs fastest ===")
    for name, t in sorted(times, key=lambda z: z[1]):
        print(f"{t/fastest:6.2f}x  | {name}")

# =========================================================================
if __name__ == "__main__":
    main()

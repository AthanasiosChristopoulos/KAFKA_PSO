import numpy as np
from sklearn.metrics import accuracy_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
from kafka import KafkaConsumer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------
# 1. Load flat weights from file
# -----------------------------
def load_flat_weights(path: str) -> np.ndarray:
    flat = np.loadtxt(path, dtype=np.float32)
    print(f"[DEBUG] Loaded flat weights from {path}, length = {len(flat)}")
    print(f"[DEBUG] First 5 flat values: {flat[:5]}")
    return flat


# ---------------------------------------------------------
# 2. Reconstruct layer weights using Dl4jParamUtils.updateModel convention
# ---------------------------------------------------------
def reconstruct_layer_weights_for_keras(flat: np.ndarray):
    layer_shapes = [
        (4, 16),    # Layer 1
        (16, 16),   # Layer 2
        (16, 3),    # Output Layer
    ]

    idx = 0
    kernels = []
    biases = []

    for layer_idx, (in_size, out_size) in enumerate(layer_shapes):
        kernel = np.zeros((in_size, out_size), dtype=np.float32)
        bias = np.zeros((out_size,), dtype=np.float32)

        for j in range(out_size):
            for i in range(in_size):
                if idx >= len(flat):
                    raise ValueError(f"Flat array too short at layer {layer_idx}, j={j}, i={i}")
                kernel[i, j] = flat[idx]
                idx += 1
            if idx >= len(flat):
                raise ValueError(f"Flat array too short when reading bias at layer {layer_idx}, j={j}")
            bias[j] = flat[idx]
            idx += 1

        kernels.append(kernel)
        biases.append(bias)

        print(f"[DEBUG] Layer {layer_idx}: kernel shape {kernel.shape}, bias shape {bias.shape}")
        print(f"[DEBUG] Layer {layer_idx}: first row of kernel: {kernel[0, :5]}, first bias: {bias[0]}")

    print(f"[DEBUG] Consumed {idx} values from flat array.")
    if idx != len(flat):
        raise ValueError(f"Flat length mismatch, used {idx} of {len(flat)} values")

    return kernels, biases


# -----------------------------
# 3. Keras model definition
# -----------------------------
def build_keras_iris_model() -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(4,)),
            layers.Dense(16, activation="relu", name="dense1"),
            layers.Dense(16, activation="relu", name="dense2"),
            layers.Dense(3, activation="softmax", name="output"),
        ]
    )
    return model


def load_weights_into_keras_model(model: keras.Model, kernels, biases):
    dense1 = model.get_layer("dense1")
    dense2 = model.get_layer("dense2")
    output = model.get_layer("output")

    dense1.set_weights([kernels[0], biases[0]])
    dense2.set_weights([kernels[1], biases[1]])
    output.set_weights([kernels[2], biases[2]])

    print("[DEBUG] Weights loaded into Keras model.")
    print("[DEBUG] dense1 kernel[0, :5] =", dense1.get_weights()[0][0, :5])
    print("[DEBUG] dense1 bias[0]       =", dense1.get_weights()[1][0])


# -----------------------------
# 4. Evaluate model using data from Kafka
# -----------------------------
def evaluate_model_from_kafka(model: keras.Model, num_samples: int = 150):
    """
    Consume `num_samples` messages from Kafka topic DATA_TOPIC.
    Each message is expected to be a JSON with keys:
      - "features": list of 4 scaled floats
      - "label": int in {0,1,2}
    """
    topic = os.getenv("DATA_TOPIC", "iris-input")
    print(f"[INFO] Consuming {num_samples} samples from Kafka topic '{topic}'")

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers="localhost:9092",
        auto_offset_reset="earliest",    # start from beginning
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="iris-eval-tf"          # separate group so we see earliest messages
    )

    X_list = []
    y_list = []

    for message in consumer:
        msg = message.value   # already deserialized JSON dict
        features = msg.get("features")
        label = msg.get("label")

        if features is None or label is None:
            continue

        X_list.append(features)
        y_list.append(int(label))

        if len(X_list) >= num_samples:
            break

    consumer.close()

    X = np.array(X_list, dtype=np.float32)   # shape [num_samples, 4]
    y = np.array(y_list, dtype=np.int64)     # shape [num_samples]

    print(f"[INFO] Collected {X.shape[0]} samples from Kafka")
    print("First 5 samples:")
    for i in range(min(5, len(X))):
        print(f"Row {i}: features={X[i].tolist()}  label={int(y[i])}")

    # These features are ALREADY STANDARDIZED by your producer (StandardScaler),
    # so we DO NOT rescale here.
    probs = model(X, training=False).numpy()   # [num_samples, 3]
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y, preds)
    print("Accuracy on Kafka Iris stream (TensorFlow/Keras):", acc)
    return acc


# -----------------------------
# 5. Main
# -----------------------------
if __name__ == "__main__":
    flat_path = "iris_streams/models/iris-weights-flat.txt"

    flat = load_flat_weights(flat_path)
    kernels, biases = reconstruct_layer_weights_for_keras(flat)

    model = build_keras_iris_model()
    load_weights_into_keras_model(model, kernels, biases)

    evaluate_model_from_kafka(model, num_samples=150)


import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # no GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from kafka import KafkaConsumer
import json

# =======================================================================================================
# 0 Set the arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--kafka', action='store_true')
args = parser.parse_args()

# =======================================================================================================
# 0. Load .env parameters

from dotenv import load_dotenv
env_path = os.path.join("..", "java", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

DATASET = os.getenv("DATASET", "iris")
NEURAL_INPUT = NEURAL_OUTPUT = 0

if(DATASET == "iris"):
    NEURAL_INPUT = int(os.getenv("NUM_FEATURES_IRIS", "4"))
    NEURAL_OUTPUT = int(os.getenv("NUM_CLASSES_IRIS", "3"))
    
elif (DATASET == "wine"):
    NEURAL_INPUT = int(os.getenv("NUM_FEATURES_WINE", "13"))
    NEURAL_OUTPUT = int(os.getenv("NUM_CLASSES_WINE", "3"))   

elif (DATASET == "mnist"):
    NEURAL_INPUT = int(os.getenv("NUM_FEATURES_MNIST", "784"))
    NEURAL_OUTPUT = int(os.getenv("NUM_CLASSES_MNIST", "10"))   
    

# =======================================================================================================
# 1. Load flat weights from file

def load_flat_weights(path: str) -> np.ndarray:
    flat = np.loadtxt(path, dtype=np.float32)
    print(f"Loaded flat weights from {path}, length = {len(flat)}")
    print(f"First 5 flat values: {flat[:5]}")
    return flat

# =======================================================================================================
# 2. Reconstruct layer weights using the same convention as

def reconstruct_layer_weights_for_keras(flat: np.ndarray):

    # (in_size, out_size) for each dense layer in order
    if DATASET == "iris":
        layer_shapes = [
            (NEURAL_INPUT, 16),     # Layer 1
            (16, 16),               # Layer 2
            (16, NEURAL_OUTPUT),    # Output Layer
        ]
        
    elif DATASET == "wine":
        layer_shapes = [
            (NEURAL_INPUT, 32),
            (32, 16),
            (16, NEURAL_OUTPUT),
        ]
        
    elif DATASET == "mnist":
        layer_shapes = [
            (NEURAL_INPUT, 256),  
            (256, 128),           
            (128, NEURAL_OUTPUT), 
        ]
                        
    else:
        raise ValueError(f"Unsupported DATASET={DATASET}")

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

    print(f"Consumed {idx} values from flat array.")
    if idx != len(flat):
        raise ValueError(f"Flat length mismatch, used {idx} of {len(flat)} values")

    return kernels, biases


# =======================================================================================================
# 3. Keras model definition

def build_keras_model():
    
    if DATASET == "iris":
        model = keras.Sequential(
            [
                layers.Input(shape=(NEURAL_INPUT,)),
                layers.Dense(16, activation="relu", name="dense1"),
                layers.Dense(16, activation="relu", name="dense2"),
                layers.Dense(NEURAL_OUTPUT, activation="softmax", name="output"),
            ]
        )
        
    elif DATASET == "wine":
        model = keras.Sequential(
            [
                layers.Input(shape=(NEURAL_INPUT,)),
                layers.Dense(32, activation="relu", name="dense1"),
                layers.Dense(16, activation="relu", name="dense2"),
                layers.Dense(NEURAL_OUTPUT, activation="softmax", name="output"),
            ]
        )
        
    elif DATASET == "mnist":
        model = keras.Sequential(
            [
                layers.Input(shape=(NEURAL_INPUT,)),   
                layers.Dense(256, activation="relu", name="dense1"),
                layers.Dense(128, activation="relu", name="dense2"),
                layers.Dense(NEURAL_OUTPUT, activation="softmax", name="output"),
            ]
        )
    else:
        raise ValueError(f"Invalid Dataset = {DATASET} chosen")
    
    return model

# =======================================================================================================
# print all weights (the entire model) to confirm

def print_full_model(kernels, biases): 

    for layer_idx, (kernel, bias) in enumerate(zip(kernels, biases), start=1):
        in_size, out_size = kernel.shape
        print(f"\n===== Layer {layer_idx} =====")
        print(f"Shape: in={in_size}, out={out_size}")

        for neuron_idx in range(out_size):
            # kernel[:, neuron_idx] is the vector of input weights for this neuron
            w_vec = kernel[:, neuron_idx]
            b_val = bias[neuron_idx]

            print(f"\n  Neuron {neuron_idx}:")
            print(f"    input weights ({len(w_vec)}): {w_vec}")
            print(f"    bias: {b_val}")

# =======================================================================================================

def load_weights_into_keras_model(model, kernels, biases):

    dense1 = model.get_layer("dense1")
    dense2 = model.get_layer("dense2")
    output = model.get_layer("output")

    dense1.set_weights([kernels[0], biases[0]])
    dense2.set_weights([kernels[1], biases[1]])
    output.set_weights([kernels[2], biases[2]])

    # input weights of neuron 0:
    print("Neuron 0 weights:", dense1.get_weights()[0][:, 0])
    print("Neuron 0 bias:", dense1.get_weights()[1][0])

    # input weights of neuron 1:
    print("Neuron 1 weights:", dense1.get_weights()[0][:, 1])
    print("Neuron 1 bias:", dense1.get_weights()[1][1])

# =======================================================================================================
# 4. Evaluate on Dataset

def evaluate_model(model):

    if DATASET == "iris":
        
        iris = load_iris()
        X_raw = iris["data"].astype(np.float32)   # shape [150, NEURAL_INPUT]
        y = iris["target"]                        # shape [150,]
        
    elif DATASET == "wine":
        iris = load_wine()
        X_raw = iris["data"].astype(np.float32)   # shape [150, NEURAL_INPUT]
        y = iris["target"]                        # shape [150,]    
    
    elif DATASET == "mnist":
        (X_train, y_train), _ = mnist.load_data()
        X_raw = X_train.reshape(-1, 28 * 28).astype("float32")
        y = y_train
        X_scaled = (X_raw / 255.0).astype(np.float32)   # no StandardScaler

    else:
        raise ValueError(f"Invalid Dataset = {DATASET} chosen")
    
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    print("=============== Showing first 5 rows of (scaled) Dataset ================")
    
    for i in range(5):
        print(f"Row {i}: features={X_scaled[i].tolist()}  label={int(y[i])}")
        
    print("================================================================")

    # model expects scaled inputs (same as Kafka training data)
    probs = model(X_scaled, training=False).numpy()
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y, preds)
    print("Test Accuracy: ", acc)
    return acc

# =======================================================================================================
# 4. Evaluate model using data from Kafka

def evaluate_model_kafka(model, num_samples: int = 150):

    data_topic = os.getenv("DATA_TOPIC", "iris-input")
    print(f"[INFO] Consuming {num_samples} samples from Kafka data_topic '{data_topic}'")

    consumer = KafkaConsumer(
        data_topic,
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

    X = np.array(X_list, dtype=np.float32)   # shape [num_samples, NEURAL_INPUT]
    y = np.array(y_list, dtype=np.int64)     # shape [num_samples]

    print(f"Collected {X.shape[0]} samples from Kafka")
    print("First 5 samples:")
    for i in range(min(5, len(X))):
        print(f"Row {i}: features={X[i].tolist()}  label={int(y[i])}")

    probs = model(X, training=False).numpy()   # [num_samples, 3]
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y, preds)
    print("Testing Accuracy on Kafka Streams: ", acc)
    return acc

# =======================================================================================================
# 5. Main

if __name__ == "__main__":

    SAVED_MODEL_NAME = os.getenv("SAVE_MODEL_NAME", "global-model")
    flat_path = os.path.join("..", "java", "models", f"{SAVED_MODEL_NAME}-flat.txt")
    
    flat = load_flat_weights(flat_path)
    kernels, biases = reconstruct_layer_weights_for_keras(flat)
    # print_full_model(kernels, biases)

    model = build_keras_model()
    load_weights_into_keras_model(model, kernels, biases)

    if args.kafka:
        evaluate_model_kafka(model)
    else:
        evaluate_model(model)

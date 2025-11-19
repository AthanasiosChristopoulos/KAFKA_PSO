# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.datasets import load_iris
# from sklearn.metrics import accuracy_score


# # -----------------------------
# # 1. Load flat weights from file
# # -----------------------------
# def load_flat_weights(path: str) -> np.ndarray:
#     """
#     Read one double per line from the file and return as a 1D numpy array.
#     """
#     flat = np.loadtxt(path, dtype=np.float32)
#     print(f"Loaded flat weights from {path}, length = {len(flat)}")
#     return flat


# # ---------------------------------------------------------
# # 2. Reconstruct layer weights using the same convention as
# #    Dl4jParamUtils.updateModel (layer-wise, j then i, then b)
# # ---------------------------------------------------------
# def reconstruct_layer_weights(flat: np.ndarray):
#     """
#     Rebuild weight matrices and bias vectors for each layer from
#     the flat array using the same indexing as Dl4jParamUtils.updateModel.

#     Architecture is:
#       - Dense 1: 4 -> 16
#       - Dense 2: 16 -> 16
#       - Dense 3: 16 -> 3
#     """
#     # (in_size, out_size) for each dense layer in order
#     layer_shapes = [
#         (4, 16),    # Layer 1
#         (16, 16),   # Layer 2
#         (16, 3),    # Output Layer
#     ]

#     idx = 0
#     weights = []
#     biases = []

#     for in_size, out_size in layer_shapes:
#         # PyTorch uses weight shape [out_features, in_features]
#         W = np.zeros((out_size, in_size), dtype=np.float32)
#         b = np.zeros((out_size,), dtype=np.float32)

#         # Java code:
#         # for j in 0..out-1:
#         #   for i in 0..in-1:
#         #       W(i,j) = flat[idx++]
#         #   b(j) = flat[idx++]
#         #
#         # We want: W_torch[j, i] = W_dl4j(i, j)
#         for j in range(out_size):
#             for i in range(in_size):
#                 W[j, i] = flat[idx]  # corresponds to W_dl4j(i, j)
#                 idx += 1
#             b[j] = flat[idx]
#             idx += 1

#         weights.append(W)
#         biases.append(b)

#     print(f"Consumed {idx} values from flat array.")
#     if idx != len(flat):
#         raise ValueError(f"Flat length mismatch, used {idx} of {len(flat)} values")

#     return weights, biases


# # -----------------------------
# # 3. PyTorch model definition
# # -----------------------------
# class IrisNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(4, 16)
#         self.fc2 = nn.Linear(16, 16)
#         self.fc3 = nn.Linear(16, 3)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)  # logits (no softmax needed for accuracy)
#         return x


# def load_weights_into_model(model: nn.Module, weights, biases):
#     """
#     Copy reconstructed numpy weights/biases into the PyTorch model.
#     """
#     with torch.no_grad():
#         model.fc1.weight.copy_(torch.from_numpy(weights[0]))
#         model.fc1.bias.copy_(torch.from_numpy(biases[0]))

#         model.fc2.weight.copy_(torch.from_numpy(weights[1]))
#         model.fc2.bias.copy_(torch.from_numpy(biases[1]))

#         model.fc3.weight.copy_(torch.from_numpy(weights[2]))
#         model.fc3.bias.copy_(torch.from_numpy(biases[2]))

#     print("Weights loaded into PyTorch model.")


# # -----------------------------
# # 4. Evaluate on Iris dataset
# # -----------------------------
# def evaluate_model(model: nn.Module):
#     """
#     Load Iris dataset from sklearn, run the model, and print accuracy.
#     """
#     iris = load_iris()
#     X = iris["data"].astype(np.float32)   # shape [150, 4]
#     y = iris["target"]                    # shape [150,]

#     X_tensor = torch.from_numpy(X)

#     model.eval()
#     with torch.no_grad():
#         logits = model(X_tensor)          # [150, 3]
#         preds = logits.argmax(dim=1).cpu().numpy()

#     acc = accuracy_score(y, preds)
#     print("Accuracy on Iris:", acc)
#     return acc


# # -----------------------------
# # 5. Main
# # -----------------------------
# if __name__ == "__main__":
#     # Path to the file written by Java when DESIRED_ACCURACY is reached
#     flat_path = "iris_streams/models/iris-weights-flat.txt"

#     flat = load_flat_weights(flat_path)
#     weights, biases = reconstruct_layer_weights(flat)

#     model = IrisNet()
#     load_weights_into_model(model, weights, biases)

#     evaluate_model(model)


import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. Load flat weights from file
# -----------------------------
def load_flat_weights(path: str) -> np.ndarray:
    """
    Read one double per line from the file and return as a 1D numpy array.
    This assumes Java wrote all values with modelToFlatList(globalModel).
    """
    flat = np.loadtxt(path, dtype=np.float32)
    print(f"[DEBUG] Loaded flat weights from {path}, length = {len(flat)}")
    print(f"[DEBUG] First 5 flat values: {flat[:5]}")
    return flat


# ---------------------------------------------------------
# 2. Reconstruct layer weights using the same convention as
#    Dl4jParamUtils.updateModel (layer-wise, j then i, then b)
# ---------------------------------------------------------
def reconstruct_layer_weights_for_keras(flat: np.ndarray):
    """
    Rebuild weight matrices and bias vectors for each layer from
    the flat array using the same indexing as Dl4jParamUtils.updateModel.

    Architecture is:
      - Dense 1: 4 -> 16
      - Dense 2: 16 -> 16
      - Dense 3: 16 -> 3

    Keras Dense kernel shape: [in_features, out_features]
    Bias shape: [out_features]
    DL4J W: [in, out], so shapes match directly (no transpose needed).
    """
    # (in_size, out_size) for each dense layer in order
    layer_shapes = [
        (4, 16),    # Layer 1
        (16, 16),   # Layer 2
        (16, 3),    # Output Layer
    ]

    idx = 0
    kernels = []
    biases = []

    for layer_idx, (in_size, out_size) in enumerate(layer_shapes):
        # Keras Dense kernel shape [in, out]
        kernel = np.zeros((in_size, out_size), dtype=np.float32)
        bias = np.zeros((out_size,), dtype=np.float32)

        # Java code (updateModel) does:
        # for j in 0..out-1:
        #   for i in 0..in-1:
        #       W.putScalar(i, j, flat[idx++]);   // W is [in, out]
        #   b.putScalar(j, flat[idx++]);
        #
        # That means flat is ordered by:
        #   for each layer:
        #     for each output neuron j:
        #       for each input i:  W(i,j)
        #       then bias(j)
        #
        # We fill kernel[i, j] = flat[...] and bias[j] accordingly.
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
    """
    Build a Keras model with the same architecture as Dl4jModelFactory.createIrisModel():
      - Dense(16, relu, input_dim=4)
      - Dense(16, relu)
      - Dense(3, softmax)
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(4,)),
            layers.Dense(16, activation="relu", name="dense1"),
            layers.Dense(16, activation="relu", name="dense2"),
            layers.Dense(3, activation="softmax", name="output"),
        ]
    )
    return model

def print_full_model(kernels, biases):
    """
    Print:
      Layer 1:
        input weights of neuron 0: [...]
        bias of neuron 0: ...
        ...
    """
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


def load_weights_into_keras_model(model: keras.Model, kernels, biases):
    """
    Copy reconstructed numpy kernel/bias arrays into the Keras model.
    """
    dense1 = model.get_layer("dense1")
    dense2 = model.get_layer("dense2")
    output = model.get_layer("output")

    dense1.set_weights([kernels[0], biases[0]])
    dense2.set_weights([kernels[1], biases[1]])
    output.set_weights([kernels[2], biases[2]])

    print("[DEBUG] Weights loaded into Keras model.")
    print("[DEBUG] dense1 kernel[0, :5] =", dense1.get_weights()[0][0, :5])
    print("[DEBUG] dense1 bias[0]       =", dense1.get_weights()[1][0])

    # input weights of neuron 0:
    print("Neuron 0 weights:", dense1.get_weights()[0][:, 0])
    print("Neuron 0 bias:", dense1.get_weights()[1][0])

    # input weights of neuron 1:
    print("Neuron 1 weights:", dense1.get_weights()[0][:, 1])
    print("Neuron 1 bias:", dense1.get_weights()[1][1])

# -----------------------------
# 4. Evaluate on Iris dataset
# -----------------------------
def evaluate_model(model: keras.Model):
    """
    Load raw Iris dataset, apply the SAME StandardScaler used by the Kafka producer,
    run the Keras model, and print accuracy.
    """
    iris = load_iris()
    X_raw = iris["data"].astype(np.float32)   # shape [150, 4]
    y = iris["target"]                        # shape [150,]

    # -----------------------------------------------------
    # Apply EXACT SAME transformation you used in producer:
    #   scaler = StandardScaler().fit(X_raw)
    #   X_scaled = scaler.transform(X_raw)
    # -----------------------------------------------------
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    print("=== Showing first 10 rows of SCALED Iris dataset (TF/Keras) ===")
    for i in range(10):
        print(
            f"Row {i}: features={X_scaled[i].tolist()}  label={int(y[i])}"
        )
    print("================================================================")

    # model expects scaled inputs (same as Kafka training data)
    probs = model(X_scaled, training=False).numpy()
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y, preds)
    print("Accuracy on Iris (TensorFlow/Keras, scaled):", acc)
    return acc

# -----------------------------
# 5. Main
# -----------------------------
if __name__ == "__main__":
    # Path to the file written by Java when DESIRED_ACCURACY is reached
    flat_path = "iris_streams/models/iris-weights-flat.txt"

    flat = load_flat_weights(flat_path)
    kernels, biases = reconstruct_layer_weights_for_keras(flat)
    print_full_model(kernels, biases)

    model = build_keras_iris_model()
    load_weights_into_keras_model(model, kernels, biases)

    evaluate_model(model)

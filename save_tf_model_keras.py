import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
 
# Load the Iris Dataset
iris = load_iris()
X = iris.data.astype("float32")   # 150 X 4, means 150 samples and 4 features
y = iris.target.astype("int32")   # 150 X 1

# Split the Dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train).astype("float32")
X_test = scaler.transform(X_test).astype("float32")

# Define Model Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),  # Input Layer, 4 because we have 4 features
    tf.keras.layers.Dense(16, activation="relu"),  # Hidden Layer 1
    tf.keras.layers.Dense(16, activation="relu"),  # Hidden layer 2
    tf.keras.layers.Dense(3, activation="softmax")  # Output Layer  
            # 3 because we have 3 classes (these are logits, score of how strongly the model leans toward each class)
])

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    # Define loss function
    
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0) 
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Validation accuracy: {acc:.3f}")

EXPORT_DIR = "iris_savedmodel"
model.export(EXPORT_DIR)            

model.save("iris_model.keras")   
print("Saved Keras model to iris_model.keras")

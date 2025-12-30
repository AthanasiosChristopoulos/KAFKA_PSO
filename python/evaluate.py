
import json, time, random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from kafka import KafkaProducer
import argparse
import numpy as np
import random
import pandas as pd
import struct
from array import array
from sklearn.preprocessing import LabelEncoder

# ========================================================================================
# Env + Args =============================================================================

from dotenv import load_dotenv
loaded = load_dotenv("../java/.env")
print("Dotenv loaded:", loaded)


print("Loading from tf.keras.datasets.mnist")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

print("Train shape:", X_train.shape, "Labels:", y_train.shape)
print("Test shape:", X_test.shape, "Labels:", y_test.shape)

class_names = [str(i) for i in range(10)]

print(X_train[1])

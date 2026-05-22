import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
import json, time, random
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
from PIL import Image
from scipy.io import loadmat
import tensorflow_datasets as tfds
import gzip

# ========================================================================================
# Env + Args =============================================================================

from dotenv import load_dotenv
loaded = load_dotenv("../java/.env")
print("Dotenv loaded:", loaded)

DATASET = os.getenv("DATASET")

PREDICTION_INPUT_TOPIC = os.getenv("PREDICTION_INPUT_TOPIC")
BATCH_FLUSH = int(os.getenv("BATCH_FLUSH"))

NUMBER_OF_DATA_REPEATS = 1
NUMBER_OF_DATA_REPEATS_TEST = 1

CNN_DATASETS = ("cifar3", "cifar5", "cifar10", "cifar10-half", "cifar5-half", 
    "nsfw", "mnist", "mnist5", "fashion-mnist", "fashion-mnist-half", "fashion-mnist5-half", "svhn", "kmnist")

parser = argparse.ArgumentParser()
parser.add_argument('--streaming', action='store_true')
parser.add_argument('--all', action='store_true')
parser.add_argument('--multi', action='store_true')
parser.add_argument('--repeat', action='store_true')
parser.add_argument('--pred', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

REPEAT = 1
if args.repeat:
    REPEAT = 10
    if("mnist" in DATASET):
        REPEAT = 3
        
# ==============================================================================================

def set_epochs():
    
    global NUMBER_OF_DATA_REPEATS, NUMBER_OF_DATA_REPEATS_TEST
    global INPUT_TOPIC, TEST_TOPIC
    
    if(DATASET == "iris" or DATASET == "wine"):     # 150 samples
        NUMBER_OF_DATA_REPEATS = 37 * 10 * REPEAT 
        NUMBER_OF_DATA_REPEATS_TEST = 3

    if(DATASET == "winequality"):
        NUMBER_OF_DATA_REPEATS = 37 * REPEAT   
        
    if(DATASET == "susy"):
        NUMBER_OF_DATA_REPEATS = 1 * REPEAT  
         
    if(DATASET == "letter"):
        NUMBER_OF_DATA_REPEATS = 20

    if(DATASET == "pendigits"):
        NUMBER_OF_DATA_REPEATS = 40 * REPEAT 

    if(DATASET == "pendigits-half"):
        NUMBER_OF_DATA_REPEATS = 80

    if(DATASET == "cifar3"):
        NUMBER_OF_DATA_REPEATS = 27

    if(DATASET == "cifar5"):
        NUMBER_OF_DATA_REPEATS = 16

    if(DATASET == "cifar10"):
        NUMBER_OF_DATA_REPEATS = 9

    if(DATASET == "cifar10-half"):
        NUMBER_OF_DATA_REPEATS = 18
        
    if(DATASET == "cifar5-half"):
        NUMBER_OF_DATA_REPEATS = 37 # its correct we are 60000 / 2 / 2 = 15000 => 15000 * 37 = 400000
        
    if(DATASET == "mnist"):
        NUMBER_OF_DATA_REPEATS = 7 * REPEAT

    if(DATASET == "mnist5"):
        NUMBER_OF_DATA_REPEATS = 7 * 2 * REPEAT
        
    if(DATASET == "kmnist"):
        NUMBER_OF_DATA_REPEATS = 7

    if(DATASET == "fashion-mnist"):
        NUMBER_OF_DATA_REPEATS = 7
    
    if(DATASET == "fashion-mnist-half"):
        NUMBER_OF_DATA_REPEATS = 14

    if(DATASET == "fashion-mnist5-half"):
        NUMBER_OF_DATA_REPEATS = 28

    if(DATASET == "svhn"):
        NUMBER_OF_DATA_REPEATS = 5
    
    print(f"REPEAT: {REPEAT}")
    print(f"NUMBER_OF_DATA_REPEATS: {NUMBER_OF_DATA_REPEATS}")
    print(f"NUMBER_OF_DATA_REPEATS_TEST: {NUMBER_OF_DATA_REPEATS_TEST}")

    INPUT_TOPIC = DATASET + "-input"
    TEST_TOPIC = DATASET + "-test"
    
# ==============================================================================================

if args.pred:
    print("Outputting to the prediction topic")
    INPUT_TOPIC = PREDICTION_INPUT_TOPIC
else:
    INPUT_TOPIC = DATASET + "-input"
    TEST_TOPIC = DATASET + "-test"

print(f"Running this on input topic: {INPUT_TOPIC}")

MAX_TEST_SAMPLES = 500

CIFAR10_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# ========================================================================================
# Kafka Producer =========================================================================

# Serializer:
def serialize_data_message(sample_index: int, features: np.ndarray, label: int) -> bytes:

    features = np.asarray(features, dtype=np.float32)
    n = int(features.size)

    header = struct.pack(">iii", int(sample_index), int(label), n)

    be = features.astype(">f4", copy=False)

    return header + be.tobytes()


kafka_host = os.getenv("KAFKA_HOST")

producer = KafkaProducer(
    bootstrap_servers=kafka_host,
    value_serializer=lambda m: serialize_data_message(m["sample_index"], m["features"], m["label"])
)

# ========================================================================================

def shuffle(X, y):

    rng = np.random.default_rng(123)   # uses a fixed rng for shuffling
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

# ========================================================================================

def evaluate_dataset(X_train, y_train, X_test, y_test, n_classes=7):
    
    if(X_train is not None and y_train is not None):
        print(
            "Train shape:", X_train.shape,
            "classes / y (labels):", (int(y_train.min()), int(y_train.max())),
            "with counts:", np.bincount(y_train, minlength=n_classes)
        )
        
    if(X_test is not None and y_test is not None):
        print(
            "Test shape:", X_test.shape,
            "classes / y (labels):", (int(y_test.min()), int(y_test.max())),
            "with counts:", np.bincount(y_test, minlength=n_classes)
        )

# ========================================================================================
# SUSY costum load function

def load_susy_sample(path, sample_size=60000):    
    
    reservoir = []

    with open(path, "r") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split(",")
            if not parts or len(parts) < 2:
                continue

            *feat_strs, label_str = parts
            features = [float(v) for v in feat_strs]
            label = int(float(label_str))  # SUSY labels are 0/1 but may appear as "0.0"

            row = (features, label)

            if len(reservoir) < sample_size:
                # fill the reservoir initially
                reservoir.append(row)
            else:
                # decide whether to replace an existing element
                j = random.randint(0, idx)
                if j < sample_size:
                    reservoir[j] = row

    X = np.array([r[0] for r in reservoir], dtype=np.float32)
    y = np.array([r[1] for r in reservoir], dtype=np.int64)
    return X, y

# ========================================================================================
# Load different datasets

def load_dataset():
    
    global NUMBER_OF_DATA_REPEATS, NUMBER_OF_DATA_REPEATS_TEST
    X = y = class_names = None
    
    # ==================================================================================================

    if DATASET == "iris":
        
        iris = load_iris()
        X, y = shuffle(iris.data, iris.target)
        class_names = iris.target_names.tolist()
        
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X).tolist() 

        evaluate_dataset(X_train, y, None, None)

        return X_scaled, y, None, None, class_names
    
    # ==================================================================================================

    elif DATASET == "mnist":
        
        print("Loading from tf.keras.datasets.mnist")
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

        X_train = X_train.astype("float32") / 255.0
        X_test  = X_test.astype("float32") / 255.0

        X_test = X_test[:MAX_TEST_SAMPLES]
        y_test = y_test[:MAX_TEST_SAMPLES]

        print("Train shape:", X_train.shape, "Labels:", y_train.shape)
        print("Test shape:", X_test.shape, "Labels:", y_test.shape)

        class_names = [str(i) for i in range(10)]

        return X_train, y_train, X_test, y_test, class_names

    # ==================================================================================================

    elif DATASET == "mnist5":
        
        print("Loading from tf.keras.datasets.mnist")
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

        X_train = X_train.astype("float32") / 255.0
        X_test  = X_test.astype("float32") / 255.0

        # Filter classes based on MNIST5 (permit only samples from the first 5 classes)
        train_mask = (y_train >= 0) & (y_train <= 4)
        test_mask  = (y_test  >= 0) & (y_test  <= 4)

        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test,  y_test  = X_test[test_mask],  y_test[test_mask]

        X_test = X_test[:MAX_TEST_SAMPLES]
        y_test = y_test[:MAX_TEST_SAMPLES]

        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)
        class_names = [str(i) for i in range(5)]

        print("Train shape:", X_train.shape, "Labels:", y_train.shape)
        print("Test shape:", X_test.shape, "Labels:", y_test.shape)
        print(f"class names: {class_names}")
        
        return X_train, y_train, X_test, y_test, class_names
        
    # ========================================================== ================

    elif DATASET == "susy":
        
        data = np.loadtxt("../data/SUSY.csv", delimiter=",", max_rows=420000)      # 400000 / 40 = 10000 per partition

        y_all = data[:, 0].astype(int)
        X_all = data[:, 1:].astype(np.float32)

        train_size = 419500 

        X_all, y_all = shuffle(X_all, y_all)

        X_train = X_all[:train_size]       
        y_train = y_all[:train_size]

        X_test = X_all[train_size:]     
        y_test = y_all[train_size:]    

        class_names = [str(i) for i in sorted(set(y_all))]

        evaluate_dataset(X_train, y_train, X_test, y_test)
        
        X_train = np.ascontiguousarray(X_train, dtype=np.float32)
        X_test  = np.ascontiguousarray(X_test, dtype=np.float32)

        return X_train, y_train, X_test, y_test, class_names
    
    # ============================================================================
    # pendigits

    elif DATASET == "pendigits":

        base_path = "../data"
        train_path = os.path.join(base_path, "my-pendigits.tra")

        print(f"Loading from: {train_path}")

        data = np.loadtxt(train_path, delimiter=",", dtype=np.float32)

        X = data[:, :-1].astype(np.float32)
        y = data[:, -1].astype(np.int64)

        X, y = shuffle(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10492, random_state=42, stratify=y)

        class_names = [str(i) for i in range(10)]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        evaluate_dataset(X_train, y_train, X_test, y_test)

        return X_train, y_train, X_test, y_test, class_names

    # ==========================================================================
    # winequality

    elif DATASET == "winequality":

        train_size = 6000       # 6000 * 37 / 40 == 5500 each. 
        random_state = 123
        path="../data/winequality.csv"
    
        print(f"Loading from: {path}")
        df = pd.read_csv(path, sep=",")

        df["type"] = df["type"].map({"white": 0, "red": 1})   

        feature_cols = [c for c in df.columns if c != "type"]

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        before = len(df)
        df = df.dropna(subset=feature_cols + ["type"]).copy()
        after = len(df)
        print(f"Dropped rows with NaNs: {before - after}")

        # y and X
        y = df["type"].astype(np.int32).values
        X = df[feature_cols].astype(np.float32).values

        X, y = shuffle(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=random_state, stratify=y)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        evaluate_dataset(X_train, y_train, X_test, y_test)

        return X_train, y_train, X_test, y_test, None

    
    # =============================================================================
    # cifar5

    elif DATASET == "cifar5":

        classes = (0, 1, 2, 3, 4)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            # X_train: (15000, 32, 32, 3), y_train: (15000,). This means 15000 * 27 = 40 * 10^4

        y_train = y_train.squeeze().astype(np.int64) 
        y_test  = y_test.squeeze().astype(np.int64)

        classes = np.array(classes, dtype=np.int64)

        train_mask = np.isin(y_train, classes)
        test_mask  = np.isin(y_test, classes)

        X_train = x_train[train_mask].astype(np.float32) / 255.0
        y_train = y_train[train_mask]
        X_test  = x_test[test_mask].astype(np.float32) / 255.0
        y_test  = y_test[test_mask]

        remap = {int(c): i for i, c in enumerate(classes.tolist())}
        y_train = np.vectorize(remap.get)(y_train).astype(np.int64)
        y_test  = np.vectorize(remap.get)(y_test).astype(np.int64)

        rng = np.random.default_rng(123)
        idx = rng.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        idx = rng.permutation(len(X_test))
        X_test, y_test = X_test[idx], y_test[idx]

        X_test = X_test[:MAX_TEST_SAMPLES]
        y_test = y_test[:MAX_TEST_SAMPLES]

        class_names = [CIFAR10_NAMES[int(c)] for c in classes]

        print(f"Selected classes: {classes.tolist()} -> {class_names}")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test : {X_test.shape},  y_test : {y_test.shape}")

        evaluate_dataset(X_train, y_train, X_test, y_test, len(class_names))

        return X_train, y_train, X_test, y_test, class_names
    
    # ================================================================================
    # cifar10

    elif DATASET == "cifar10":

        classes = np.arange(10, dtype=np.int64)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        y_train = y_train.squeeze().astype(np.int64)
        y_test  = y_test.squeeze().astype(np.int64)

        X_train = x_train.astype(np.float32) / 255.0
        X_test  = x_test.astype(np.float32) / 255.0

        rng = np.random.default_rng(123)
        idx = rng.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        idx = rng.permutation(len(X_test))
        X_test, y_test = X_test[idx], y_test[idx]

        X_test = X_test[:MAX_TEST_SAMPLES]
        y_test = y_test[:MAX_TEST_SAMPLES]

        class_names = [CIFAR10_NAMES[int(c)] for c in classes]

        print(f"Selected classes: {classes.tolist()} -> {class_names}")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test : {X_test.shape},  y_test : {y_test.shape}")

        evaluate_dataset(X_train, y_train, X_test, y_test, len(class_names))

        return X_train, y_train, X_test, y_test, class_names

    # ======================================================================================

    elif DATASET == "cifar10-half":

        classes = np.arange(10, dtype=np.int64)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        y_train = y_train.squeeze().astype(np.int64)
        y_test  = y_test.squeeze().astype(np.int64)

        X_train = x_train.astype(np.float32) / 255.0
        X_test  = x_test.astype(np.float32) / 255.0

        # split dataset in a deterministic manner (specify rng)
        rng = np.random.default_rng(123)

        idx = rng.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        half = len(X_train) // 2

        X_train_hist = X_train[:half]
        y_train_hist = y_train[:half]
        
        idx_hist = rng.permutation(len(X_train_hist))
        X_train_hist = X_train_hist[idx_hist]
        y_train_hist = y_train_hist[idx_hist]

        idx = rng.permutation(len(X_test))
        X_test, y_test = X_test[idx], y_test[idx]

        X_test = X_test[:MAX_TEST_SAMPLES]
        y_test = y_test[:MAX_TEST_SAMPLES]

        class_names = [CIFAR10_NAMES[int(c)] for c in classes]

        print(f"Selected classes: {classes.tolist()} -> {class_names}")
        print(f"Historic train: {X_train_hist.shape}, {y_train_hist.shape}")
        print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

        evaluate_dataset(X_train_hist, y_train_hist, X_test, y_test, len(class_names))

        return (
            X_train_hist, y_train_hist,
            X_test, y_test,
            class_names
        )

    # ==================================================================================================
        
    elif DATASET == "cifar5-half":

        classes = np.array((0, 1, 4, 8, 9), dtype=np.int64) # airplane, automobile, deer, ship, truck

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        y_train = y_train.squeeze().astype(np.int64)
        y_test  = y_test.squeeze().astype(np.int64)

        X_train = x_train.astype(np.float32) / 255.0
        X_test  = x_test.astype(np.float32) / 255.0

        # split dataset in a deterministic manner (specify rng)

        rng = np.random.default_rng(123) 

        idx = rng.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        half = len(X_train) // 2

        X_train_hist = X_train[:half]
        y_train_hist = y_train[:half]

        # Filter classes based on CIFAR5 (permit only samples from the selected 5 classes)
        train_mask = np.isin(y_train_hist, classes)
        test_mask  = np.isin(y_test, classes)

        X_train_hist = X_train_hist[train_mask]
        y_train_hist = y_train_hist[train_mask]

        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        label_map = {0: 0, 1: 1, 4: 2, 8: 3, 9: 4}

        y_train_hist = np.array([label_map[int(y)] for y in y_train_hist], dtype=np.int64)
        y_test       = np.array([label_map[int(y)] for y in y_test], dtype=np.int64)

        idx_hist = rng.permutation(len(X_train_hist))
        X_train_hist = X_train_hist[idx_hist]
        y_train_hist = y_train_hist[idx_hist]

        idx_test = rng.permutation(len(X_test))
        X_test = X_test[idx_test]
        y_test = y_test[idx_test]

        X_test = X_test[:MAX_TEST_SAMPLES]
        y_test = y_test[:MAX_TEST_SAMPLES]

        class_names = [CIFAR10_NAMES[int(c)] for c in classes]

        print(f"Selected classes: {classes.tolist()} -> {class_names}")
        print(f"Historic CIFAR5 train: {X_train_hist.shape}, {y_train_hist.shape}")
        print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")
        print(f"Remapped labels in train: {np.unique(y_train_hist)}")
        print(f"Remapped labels in test : {np.unique(y_test)}")

        evaluate_dataset(X_train_hist, y_train_hist, X_test, y_test, len(class_names))

        return (
            X_train_hist, y_train_hist,
            X_test, y_test,
            class_names
        )
        
    # ==================================================================================================
    # svhn

    elif DATASET == "svhn":

        classes = np.arange(10, dtype=np.int64)

        svhn_dir = "../data/svhn"
        train_path = os.path.join(svhn_dir, "train_32x32.mat")
        test_path  = os.path.join(svhn_dir, "test_32x32.mat")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing SVHN file: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing SVHN file: {test_path}")

        train_data = loadmat(train_path)
        test_data  = loadmat(test_path)

        x_train = np.transpose(train_data["X"], (3, 0, 1, 2))
        y_train = train_data["y"].reshape(-1)

        x_test = np.transpose(test_data["X"], (3, 0, 1, 2))
        y_test = test_data["y"].reshape(-1)

        y_train[y_train == 10] = 0
        y_test[y_test == 10] = 0

        y_train = y_train.astype(np.int64)
        y_test  = y_test.astype(np.int64)

        X_train = x_train.astype(np.float32) / 255.0
        X_test  = x_test.astype(np.float32) / 255.0

        rng = np.random.default_rng(123)
        idx = rng.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        idx = rng.permutation(len(X_test))
        X_test, y_test = X_test[idx], y_test[idx]
        
        X_test = X_test[:MAX_TEST_SAMPLES]
        y_test = y_test[:MAX_TEST_SAMPLES]

        class_names = [str(int(c)) for c in classes]

        print(f"Selected classes: {classes.tolist()} -> {class_names}")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test : {X_test.shape},  y_test : {y_test.shape}")

        evaluate_dataset(X_train, y_train, X_test, y_test, len(class_names))

        return X_train, y_train, X_test, y_test, class_names
    
    # ==================================================================================================
    # No datasets chosen / enviromental variable is wrong
    else:
        print("Invalid Dataset selected")
        exit(0)
        
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X).tolist()
    
    return X_scaled, y, None, None, class_names


# ==================================================================================================

def main():
    
    global DATASET
    
    if args.multi:
        # DATASET_MULTI = ["winequality", "mnist5", "mnist", "cifar5-half"]
        # DATASET_MULTI = ["winequality", "pendigits"]
        DATASET_MULTI = ["cifar5-half", "winequality", "mnist5"]
        print(f"Multi Dataset with: {DATASET_MULTI}" )
    else:
        DATASET_MULTI = [DATASET]
    
    for dataset_name in DATASET_MULTI:
        DATASET = dataset_name

        set_epochs()
          
        X_train, y_train, X_test, y_test, class_names = load_dataset()

        load_training_data = True
        load_test_data = True

        if(args.train):
            load_test_data = False

        if(args.test):
            load_training_data = False

        index = 0
        data_repeats = 0
        
        if args.eval:
            exit(0)
            
        # =================================================================

        if args.all:   
            
            # Load to Training Topic =====================================    
            
            if(load_training_data):
                
                while data_repeats < NUMBER_OF_DATA_REPEATS:
                    
                    for index in range(len(X_train)):
                        
                        if(DATASET in CNN_DATASETS):
                            
                            features = X_train[index].ravel().astype(np.float32)    # This is float type
                            # NHWC interleaved: X_train[index] has shape (32, 32, 3) (NHWC image)
                            # .ravel() in C-order flattens the last axis fastest
                                # if its (32, 32, 3) => last axis is the channel axis, then the columns axis
                                # if its (28, 28, 1) => this is row major, column axis change faster
                            # (row, col, channel) with channel changing fastest
                            # (0,0,0), (0,0,1), (0,0,2)
                            # then next pixel (0,1,0), (0,1,1), (0,1,2)
                            
                        else:
                            features = X_train[index]

                        label = int(y_train[index])
        
                        msg = {
                            "sample_index": index,
                            "features": features,
                            "label": label
                        }

                        producer.send(INPUT_TOPIC, value=msg)   # this has key=None ... essentially this means use round robin for partitioning the records 
                        
                        if index % BATCH_FLUSH == 0:
                            producer.flush()

                    producer.flush()

                    data_repeats += 1
                
                print(f"Loaded entire {DATASET} dataset in {INPUT_TOPIC}")
            
            # Load to Test Topic =====================================================================
                    
            if(load_test_data):
                    
                data_repeats = 0
                
                if X_test is not None and y_test is not None:     
                    while data_repeats < NUMBER_OF_DATA_REPEATS_TEST:
                        
                        for index in range(len(X_test)):

                            if(DATASET in CNN_DATASETS):
                                features = X_test[index].ravel().astype(np.float32)
                            else:
                                features = X_test[index]
                                
                            label = int(y_test[index])

                            msg = {
                                "sample_index": index,
                                "features": features,
                                "label": label
                            }
                            
                            key = f"{data_repeats}:{index}"
                            producer.send(TEST_TOPIC, key=key.encode("utf-8"), value=msg)

                            if index % BATCH_FLUSH == 0:
                                producer.flush()
                        
                        producer.flush() 
                        data_repeats += 1
                                    
                    print(f"Loaded entire {DATASET} dataset in {TEST_TOPIC}")
                
                # Load to Test Topic with Training Data =========================================

                else:   # If there are no explicit test samples then load the training samples in the test topic (for iris)
                    
                    while data_repeats < NUMBER_OF_DATA_REPEATS_TEST:
                        
                        for index in range(len(X_train)):

                            if(DATASET in CNN_DATASETS):
                                features = X_train[index].ravel().astype(np.float32)
                            else:
                                features = X_train[index]

                            label = int(y_train[index])

                            msg = {
                                "sample_index": index,
                                "features": features,
                                "label": label
                            }

                            key = f"{data_repeats}:{index}"
                            producer.send(TEST_TOPIC, key=key.encode("utf-8"), value=msg)

                            if index % BATCH_FLUSH == 0:
                                producer.flush()

                        producer.flush() 
                        data_repeats += 1
                                    
                    print(f"Loaded entire {DATASET} dataset in {TEST_TOPIC}")
        
        # ===================================================================

        else:
            print("Arg missing")

# ===================================================================================

if __name__ == "__main__":
    main()
                              
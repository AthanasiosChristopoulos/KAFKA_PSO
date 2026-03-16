import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
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
    "nsfw", "mnist", "mnist5", "fashion-mnist", "svhn", "kmnist")

parser = argparse.ArgumentParser()
parser.add_argument('--streaming', action='store_true')
parser.add_argument('--all', action='store_true') # make this a flag argument
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
    if(DATASET == "mnist"):
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
        NUMBER_OF_DATA_REPEATS = 37
        
    if(DATASET == "mnist"):
        NUMBER_OF_DATA_REPEATS = 7 * REPEAT

    if(DATASET == "mnist5"):
        NUMBER_OF_DATA_REPEATS = 15
        
    if(DATASET == "kmnist"):
        NUMBER_OF_DATA_REPEATS = 7

    if(DATASET == "fashion-mnist"):
        NUMBER_OF_DATA_REPEATS = 7
        
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
    # INPUT_TOPIC = DATA_TOPIC
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

def _read_idx_images_gz(path):
    with gzip.open(path, "rb") as f:
        data = f.read()

    magic = int.from_bytes(data[0:4], "big")
    if magic != 2051:
        raise ValueError(f"Invalid image file magic number in {path}: {magic}")

    num_images = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")

    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    images = images.reshape(num_images, rows, cols)
    return images

# =======================================================================================

def _read_idx_labels_gz(path):
    with gzip.open(path, "rb") as f:
        data = f.read()

    magic = int.from_bytes(data[0:4], "big")
    if magic != 2049:
        raise ValueError(f"Invalid label file magic number in {path}: {magic}")

    num_labels = int.from_bytes(data[4:8], "big")
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    labels = labels.reshape(num_labels)
    return labels

# =======================================================================================

def load_kmnist(data_dir):

    train_images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    test_images_path  = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path  = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

    required_files = [
        train_images_path,
        train_labels_path,
        test_images_path,
        test_labels_path,
    ]

    missing = [p for p in required_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "KMNIST files not found.\n"
            "Please place these files in:\n"
            f"  {data_dir}\n\n"
            "Missing:\n  " + "\n  ".join(missing)
        )

    x_train = _read_idx_images_gz(train_images_path)
    y_train = _read_idx_labels_gz(train_labels_path)
    x_test  = _read_idx_images_gz(test_images_path)
    y_test  = _read_idx_labels_gz(test_labels_path)

    # normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # add channel dim -> (N, 28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test  = np.expand_dims(x_test, axis=-1)

    y_train = y_train.astype("int32")
    y_test  = y_test.astype("int32")

    print("KMNIST loaded from local files:")
    print("  x_train:", x_train.shape, x_train.dtype)
    print("  y_train:", y_train.shape, y_train.dtype)
    print("  x_test :", x_test.shape, x_test.dtype)
    print("  y_test :", y_test.shape, y_test.dtype)

    return x_train, y_train, x_test, y_test

# ========================================================================================

def load_susy_sample(path, sample_size=60000):       # shuffle large files (pick random values)
    
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

def shuffle(X, y):

    rng = np.random.default_rng(123)   # uses a fixed field for shuffling
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

# ========================================================================================

def evaluate_dataset(X_train, y_train, X_test, y_test, n_classes=7):
    
    if(X_train != None and y_train != None):
        print(
            "Train shape:", X_train.shape,
            "classes / y (labels):", (int(y_train.min()), int(y_train.max())),
            "with counts:", np.bincount(y_train, minlength=n_classes)
        )
        
    if(X_test != None and y_test != None):
        print(
            "Test shape:", X_test.shape,
            "classes / y (labels):", (int(y_test.min()), int(y_test.max())),
            "with counts:", np.bincount(y_test, minlength=n_classes)
        )

# ========================================================================================

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

    elif DATASET == "wine":  

        wine = load_wine()
        X, y = shuffle(wine.data, wine.target)
        # X = wine.data              # shape (178, 13)
        # y = wine.target            # 0,1,2
        class_names = wine.target_names.tolist()
        
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X).tolist() 
    
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
    
    # ==================================================================================================

    elif DATASET == "fashion-mnist":
        
        print("Loading from tf.keras.datasets.fashion-mnist")
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

        # Normalize to [0,1]
        X_train = X_train.astype("float32") / 255.0
        X_test  = X_test.astype("float32") / 255.0

        # Optionally clip test size (same as you do for MNIST)
        X_test = X_test[:MAX_TEST_SAMPLES]
        y_test = y_test[:MAX_TEST_SAMPLES]

        print("Train shape:", X_train.shape, "Labels:", y_train.shape)
        print("Test shape:", X_test.shape, "Labels:", y_test.shape)

        class_names = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot"
        ]

        return X_train, y_train, X_test, y_test, class_names

    # ==================================================================================================
    elif DATASET == "kmnist":

        print("Loading KMNIST from local NAS files")

        X_train, y_train, X_test, y_test = load_kmnist(data_dir="/mnt/nas_drive/achristopoulos/KAFKA_PSO_4/data/kmnist")

        X_test = X_test[:MAX_TEST_SAMPLES]
        y_test = y_test[:MAX_TEST_SAMPLES]

        print("Train shape:", X_train.shape, "Labels:", y_train.shape)
        print("Test shape:", X_test.shape, "Labels:", y_test.shape)

        class_names = [
            "o", "ki", "su", "tsu", "na",
            "ha", "ma", "ya", "re", "wo"
        ]

        return X_train, y_train, X_test, y_test, class_names
        
    # ==================================================================================================

    elif DATASET == "susy":
        
        # data = np.loadtxt("../data/SUSY.csv", delimiter=",", max_rows=80000) # 80000 - 100000
                                                                             # (5000000, 19), the 19th is the label
                                                                            #  5000000
                                                                            #    80000
                                                                            #  3200000
                                                                            #   120000
        # data = np.loadtxt("../data/SUSY.csv", delimiter=",", max_rows=80000, skiprows=80000)
        # data = np.loadtxt("../data/SUSY.csv", delimiter=",", max_rows=2420000)    # 60000 per partition

        data = np.loadtxt("../data/SUSY.csv", delimiter=",", max_rows=420000)      # 400000 / 40 = 10000 per partition

        y_all = data[:, 0].astype(int)
        X_all = data[:, 1:].astype(np.float32)

        # ===== Train/Test Split =====
        # train_size = 2400000      # there are approximately 60000 messages inside each partition (40 partitions in topic)
        #                           # the test size is 20000
        
        # train_size = 400000         # there are approximately 10000 messages inside each partition (40 partitions in topic)
                                    # the test size is 20000

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

    # ==================================================================================================

    elif DATASET == "bank":
        
        data = np.loadtxt("../data/processed_bank.csv", delimiter=",", dtype=np.float32, skiprows=1)

        y_all = data[:, -1].astype(int)

        X_all = data[:, :-1].astype(np.float32)

        # ===== Train/Test Split =====
        train_size = 30000    # adjust as you want

        X_all, y_all = shuffle(X_all, y_all)

        X_train = X_all[:train_size]        # training features
        y_train = y_all[:train_size]        # training labels

        X_test = X_all[train_size:]   # test features
        y_test = y_all[train_size:]   # test labels

        class_names = [str(i) for i in sorted(set(y_all))]

        evaluate_dataset(X_train, y_train, X_test, y_test)
        
        return X_train.tolist(), y_train, X_test.tolist(), y_test, class_names

    # ====================================================================================================

    elif DATASET == "adult":
        
        cols = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country",
            "income"
        ]

        # ===== Load TRAIN from adult.data =====
        df_train = pd.read_csv("../data/adult.data", header=None, names=cols, sep=",",engine="python",skipinitialspace=True)

        # ===== Load TEST from adult.test =====
        df_test = pd.read_csv("../data/adult.test", header=None, names=cols, sep=",",
            engine="python",skipinitialspace=True,skiprows=1)

        df_test["income"] = df_test["income"].astype(str).str.replace(".", "", regex=False)

        df_train.replace("?", np.nan, inplace=True)
        df_test.replace("?", np.nan, inplace=True)
        df_train.dropna(inplace=True)
        df_test.dropna(inplace=True)

        y_train = (df_train["income"] == ">50K").astype(int).values
        y_test = (df_test["income"] == ">50K").astype(int).values

        X_train_df = df_train.drop(columns=["income"])
        X_test_df  = df_test.drop(columns=["income"])

        X_train_oh = pd.get_dummies(X_train_df, drop_first=True)
        X_test_oh  = pd.get_dummies(X_test_df, drop_first=True)

        X_test_oh = X_test_oh.reindex(columns=X_train_oh.columns, fill_value=0)

        X_train_raw = X_train_oh.astype(np.float32).values
        X_test_raw  = X_test_oh.astype(np.float32).values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test  = scaler.transform(X_test_raw).astype(np.float32)

        class_names = ["<=50K", ">50K"]

        evaluate_dataset(X_train, y_train, X_test, y_test)

        return X_train.tolist(), y_train, X_test.tolist(), y_test, class_names

    # ====================================================================================================
    
    elif DATASET == "covertype":

        path = "../data/covertype.csv"
        label_col = "Cover_Type"

        print(f"Loading from: {path}")

        df = pd.read_csv(path, nrows=80000)
        print("Raw shape:", df.shape)

        y_all = df[label_col].astype(np.int64).to_numpy()                 
        X_all = df.drop(columns=[label_col]).astype(np.float32).to_numpy()

        y_all = y_all - 1

        X_all, y_all = shuffle(X_all, y_all)

        # ===== Train/Test Split (80/20) =====
        train_size = int(0.8 * len(y_all))

        X_train_raw = X_all[:train_size]
        y_train = y_all[:train_size]

        X_test_raw = X_all[train_size:]
        y_test = y_all[train_size:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test  = scaler.transform(X_test_raw).astype(np.float32)

        class_names = [str(i) for i in range(7)]  # "0".."6"

        evaluate_dataset(X_train, y_train, X_test, y_test)

        return X_train.tolist(), y_train, X_test.tolist(), y_test, class_names
    
    # ====================================================================================================

    elif DATASET == "har":
        
        base_path="../data/"
        
        train_x_path = os.path.join(base_path, "X_train.txt")
        train_y_path = os.path.join(base_path, "y_train.txt")
        test_x_path  = os.path.join(base_path, "X_test.txt")
        test_y_path  = os.path.join(base_path, "y_test.txt")

        print(f"[HAR] Loading from: {base_path}")
        X_train = np.loadtxt(train_x_path, dtype=np.float32)
        y_train = np.loadtxt(train_y_path, dtype=np.int64)
        X_test  = np.loadtxt(test_x_path,  dtype=np.float32)
        y_test  = np.loadtxt(test_y_path,  dtype=np.int64)

        y_train = y_train - 1   # 1 ... 6 -> 0 ... 5
        y_test  = y_test - 1

        class_names = [str(i) for i in range(6)]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        evaluate_dataset(X_train, y_train, X_test, y_test)

        return X_train.tolist(), y_train, X_test.tolist(), y_test, class_names

    # ====================================================================================================
    # pendigits

    elif DATASET == "pendigits":

        base_path = "../data"
        train_path = os.path.join(base_path, "my-pendigits.tra")

        print(f"Loading from: {train_path}")

        data = np.loadtxt(train_path, delimiter=",", dtype=np.float32)

        X = data[:, :-1].astype(np.float32)   # (n, 16)
        y = data[:, -1].astype(np.int64)      # (n,)

        X, y = shuffle(X, y)

        # Split manually
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=10492, random_state=42, stratify=y)

        class_names = [str(i) for i in range(10)]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        evaluate_dataset(X_train, y_train, X_test, y_test)

        return X_train, y_train, X_test, y_test, class_names


    # ====================================================================================================
    # pendigits-half

    elif DATASET == "pendigits-half":

        base_path = "../data"
        train_path = os.path.join(base_path, "my-pendigits.tra")

        print(f"Loading from: {train_path}")

        data = np.loadtxt(train_path, delimiter=",", dtype=np.float32)

        X = data[:, :-1].astype(np.float32)   # (n, 16)
        y = data[:, -1].astype(np.int64)      # (n,)

        mask = y < 5

        X = X[mask]
        y = y[mask]
        
        X, y = shuffle(X, y)

        # Split manually
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5246, random_state=42, stratify=y)

        class_names = [str(i) for i in range(5)]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        evaluate_dataset(X_train, y_train, X_test, y_test)

        return X_train, y_train, X_test, y_test, class_names
    
    # ====================================================================================================
    # winequality

    elif DATASET == "winequality":

        train_size = 6000       # 6000 * 37 / 40 == 5500 each. 
        random_state = 123
        path="../data/winequality.csv"
    
        print(f"Loading from: {path}")
        df = pd.read_csv(path, sep=",")

        df["type"] = df["type"].map({"white": 0, "red": 1})    # Map label: white=0, red=1

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

        # Standardize  ==============================================================
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        evaluate_dataset(X_train, y_train, X_test, y_test)

        return X_train, y_train, X_test, y_test, None

    # ==================================================================================================
    # letter

    elif DATASET == "letter":
    
        df = pd.read_csv("../data/letter-recognition.csv")
        X = df.drop("letter", axis=1).values
        y = df["letter"].values
        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)
        train_size = 19500
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, train_size=train_size, random_state=42, stratify=y_enc)
        evaluate_dataset(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test, None

    # ==================================================================================================
    # cifar3

    elif DATASET == "cifar3":

        classes = (0,1,2)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            # X_train: (15000, 32, 32, 3), y_train: (15000,). This means 15000 * 27 = 40 * 10^4

        y_train = y_train.squeeze().astype(np.int64)  # (N,)
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
    
    # ==================================================================================================
    # cifar5

    elif DATASET == "cifar5":

        classes = (0, 1, 2, 3, 4)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            # X_train: (15000, 32, 32, 3), y_train: (15000,). This means 15000 * 27 = 40 * 10^4

        y_train = y_train.squeeze().astype(np.int64)  # (N,)
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
    
    # ==================================================================================================
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

    # ==================================================================================================

    elif DATASET == "cifar10-half":

        classes = np.arange(10, dtype=np.int64)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        y_train = y_train.squeeze().astype(np.int64)
        y_test  = y_test.squeeze().astype(np.int64)

        X_train = x_train.astype(np.float32) / 255.0
        X_test  = x_test.astype(np.float32) / 255.0

        rng = np.random.default_rng(123)

        # -------------------------------------------------
        # Step 1: create reproducible split
        # -------------------------------------------------
        
        idx = rng.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        half = len(X_train) // 2

        # So it takes the first / lower / earlier half of the shuffled CIFAR-10 training set.
        X_train_hist = X_train[:half]
        y_train_hist = y_train[:half]
        
        # -------------------------------------------------
        # Step 2: shuffle each half independently
        # -------------------------------------------------

        idx_hist = rng.permutation(len(X_train_hist))
        X_train_hist = X_train_hist[idx_hist]
        y_train_hist = y_train_hist[idx_hist]

        # -------------------------------------------------
        # shuffle test set
        # -------------------------------------------------

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

        classes = np.array((0, 1, 4, 8, 9), dtype=np.int64)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        y_train = y_train.squeeze().astype(np.int64)
        y_test  = y_test.squeeze().astype(np.int64)

        X_train = x_train.astype(np.float32) / 255.0
        X_test  = x_test.astype(np.float32) / 255.0

        rng = np.random.default_rng(123)

        # -------------------------------------------------
        # Step 1: reproducible split of the full training set
        # -------------------------------------------------
        idx = rng.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        half = len(X_train) // 2

        X_train_hist = X_train[:half]
        y_train_hist = y_train[:half]

        # -------------------------------------------------
        # Step 2: keep only selected CIFAR-5 classes
        # -------------------------------------------------
        train_mask = np.isin(y_train_hist, classes)
        test_mask  = np.isin(y_test, classes)

        X_train_hist = X_train_hist[train_mask]
        y_train_hist = y_train_hist[train_mask]

        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        # -------------------------------------------------
        # Step 3: remap labels {0,1,4,8,9} -> {0,1,2,3,4}
        # -------------------------------------------------
        label_map = {0: 0, 1: 1, 4: 2, 8: 3, 9: 4}

        y_train_hist = np.array([label_map[int(y)] for y in y_train_hist], dtype=np.int64)
        y_test       = np.array([label_map[int(y)] for y in y_test], dtype=np.int64)

        # -------------------------------------------------
        # Step 4: shuffle the filtered training half
        # -------------------------------------------------
        idx_hist = rng.permutation(len(X_train_hist))
        X_train_hist = X_train_hist[idx_hist]
        y_train_hist = y_train_hist[idx_hist]

        # -------------------------------------------------
        # Step 5: shuffle / trim test
        # -------------------------------------------------
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

    elif DATASET == "nsfw":

        NSFW_CLASS_NAMES = ["drawings", "hentai", "neutral", "porn", "sexy"]
        TEST_SAMPLES = 500
        IMAGE_SIZE = 32
        
        if IMAGE_SIZE not in (32, 64):
            raise ValueError("For NSFW, IMAGE_SIZE must be 32 or 64")

        DATA_DIR = "../data/nsfw_dataset_v1"

        def load_and_preprocess_image(img_path, image_size):
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize((image_size, image_size), Image.BILINEAR)
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                return arr
            except Exception as e:
                print(f"[WARNING] Skipping corrupt image: {img_path} -> {e}")
                return None

        X_all = []
        y_all = []

        class_names = []
        class_to_idx = {}

        # Read classes in fixed order
        for class_idx, class_name in enumerate(NSFW_CLASS_NAMES):
            class_dir = os.path.join(dataset_dir, class_name)

            if not os.path.isdir(class_dir):
                print(f"[WARNING] Missing class folder: {class_dir}")
                continue

            class_names.append(class_name)
            class_to_idx[class_name] = len(class_names) - 1

            filenames = sorted(os.listdir(class_dir))

            for fname in filenames:
                fpath = os.path.join(class_dir, fname)

                if not os.path.isfile(fpath):
                    continue

                lower = fname.lower()
                if not lower.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    continue

                arr = load_and_preprocess_image(fpath, IMAGE_SIZE)
                if arr is None:
                    continue

                X_all.append(arr)
                y_all.append(class_to_idx[class_name])

        if len(X_all) == 0:
            raise ValueError(f"No images were loaded from dataset_dir={dataset_dir}")

        X_all = np.asarray(X_all, dtype=np.float32)
        y_all = np.asarray(y_all, dtype=np.int64)

        rng = np.random.default_rng(123)
        idx = rng.permutation(len(X_all))
        X_all = X_all[idx]
        y_all = y_all[idx]

        if len(X_all) < TEST_SAMPLES:
            raise ValueError(
                f"Dataset has only {len(X_all)} samples, cannot create test set of {TEST_SAMPLES}"
            )

        X_test = X_all[:TEST_SAMPLES]
        y_test = y_all[:TEST_SAMPLES]

        X_train = X_all[TEST_SAMPLES:]
        y_train = y_all[TEST_SAMPLES:]

        print(f"Selected classes: {class_names}")
        print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")

        # Optional label distribution print
        train_counts = np.bincount(y_train, minlength=len(class_names))
        test_counts = np.bincount(y_test, minlength=len(class_names))

        print("Train distribution:")
        for i, name in enumerate(class_names):
            print(f"  {name:10s}: {train_counts[i]}")

        print("Test distribution:")
        for i, name in enumerate(class_names):
            print(f"  {name:10s}: {test_counts[i]}")

        evaluate_dataset(X_train, y_train, X_test, y_test, len(class_names))

        return X_train, y_train, X_test, y_test, class_names
    
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

        # SVHN format:
        # X: (32, 32, 3, N)
        # y: (N, 1), where label 10 means digit 0

        x_train = np.transpose(train_data["X"], (3, 0, 1, 2))
        y_train = train_data["y"].reshape(-1)

        x_test = np.transpose(test_data["X"], (3, 0, 1, 2))
        y_test = test_data["y"].reshape(-1)

        # Convert label 10 -> 0
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
        # DATASET_MULTI = ["iris", "winequality", "mnist5", "mnist", "cifar5-half", "kmnist", "fashion-mnist"]
        # DATASET_MULTI = ["kmnist", "fashion-mnist"]
        DATASET_MULTI = ["iris", "winequality", "pendigits"]
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
            
        # =================================================================================================================

        if args.all:   
            
            # Load to Training Topic ==================================================================        
            
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
                
                # ==========================================================================================================

                else:   # If there are no explicit training samples then load the training samples in the test topic
                    
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
        
        # =================================================================================================================
        
        elif args.streaming:
            
            while True:
                index = random.randrange(len(X_train))      # we need random samples (if in order, they would belong to the same class)
                features = X_train[index]
                label = int(y_train[index])   

                msg = { 
                    "sample_index" : index,
                    "features": features,
                    "label": label,
                } 

                producer.send(INPUT_TOPIC, value=msg)       # Kafka Producer doesnt send immidiately, it buffers messages into a queue and sends them in batches
                producer.flush()                            # This sends everything that been buffered

                print(f"sent: {msg}")
                index = index + 1
                
                time.sleep(1)

        else:
            print("Arg missing")

# ===================================================================================

if __name__ == "__main__":
    main()
                              

import json, time, random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

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

DATASET = os.getenv("DATASET")

PREDICTION_INPUT_TOPIC = os.getenv("PREDICTION_INPUT_TOPIC")
BATCH_FLUSH = int(os.getenv("BATCH_FLUSH"))

NUMBER_OF_DATA_REPEATS = 1
NUMBER_OF_DATA_REPEATS_TEST = 1

if(DATASET == "iris" or DATASET == "wine"):
    NUMBER_OF_DATA_REPEATS = 2500
    NUMBER_OF_DATA_REPEATS_TEST = 2

if(DATASET == "winequality"):
    NUMBER_OF_DATA_REPEATS = 37     
    NUMBER_OF_DATA_REPEATS_TEST = 1

if(DATASET == "letter"):
    NUMBER_OF_DATA_REPEATS = 20
    NUMBER_OF_DATA_REPEATS_TEST = 1

if(DATASET == "pendigits"):
    NUMBER_OF_DATA_REPEATS = 40
    NUMBER_OF_DATA_REPEATS_TEST = 1

if(DATASET == "pendigits-half"):
    NUMBER_OF_DATA_REPEATS = 80
    NUMBER_OF_DATA_REPEATS_TEST = 1

if(DATASET == "cifar3"):
    NUMBER_OF_DATA_REPEATS = 27
    NUMBER_OF_DATA_REPEATS_TEST = 1

if(DATASET == "mnist"):
    NUMBER_OF_DATA_REPEATS = 7
    NUMBER_OF_DATA_REPEATS_TEST = 1

if(DATASET == "mnist4"):
    NUMBER_OF_DATA_REPEATS = 17
    NUMBER_OF_DATA_REPEATS_TEST = 1

print(f"NUMBER_OF_DATA_REPEATS: {NUMBER_OF_DATA_REPEATS}")
print(f"NUMBER_OF_DATA_REPEATS_TEST: {NUMBER_OF_DATA_REPEATS_TEST}")

parser = argparse.ArgumentParser()
parser.add_argument('--streaming', action='store_true')
parser.add_argument('--all', action='store_true') # make this a flag argument
parser.add_argument('--pred', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

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

# producer = KafkaProducer(
#     bootstrap_servers = "localhost:9092",
#     value_serializer = lambda v: json.dumps(v).encode("utf-8") # convert json int bytes before sending
# )

# .toBytes():
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda m: serialize_data_message(m["sample_index"], m["features"], m["label"])
)

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
    
    print(
        "Train shape:", X_train.shape,
        "classes / y (labels):", (int(y_train.min()), int(y_train.max())),
        "with counts:", np.bincount(y_train, minlength=n_classes)
    )
    print(
        "Test shape:", X_test.shape,
        "classes / y (labels):", (int(y_test.min()), int(y_test.max())),
        "with counts:", np.bincount(y_test, minlength=n_classes)
    )

# ========================================================================================

def load_dataset():
    
    global NUMBER_OF_DATA_REPEATS, NUMBER_OF_DATA_REPEATS_TEST
    X = y = class_names = None
    
    if DATASET == "iris":
        iris = load_iris()
        X, y = shuffle(iris.data, iris.target)
        class_names = iris.target_names.tolist()
        
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X).tolist() 
    
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

    elif DATASET == "mnist4":
        
        print("Loading from tf.keras.datasets.mnist")
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

        X_train = X_train.astype("float32") / 255.0
        X_test  = X_test.astype("float32") / 255.0

        train_mask = (y_train >= 0) & (y_train <= 3)
        test_mask  = (y_test  >= 0) & (y_test  <= 3)

        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test,  y_test  = X_test[test_mask],  y_test[test_mask]

        X_test = X_test[:MAX_TEST_SAMPLES]
        y_test = y_test[:MAX_TEST_SAMPLES]

        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)

        print("Train shape:", X_train.shape, "Labels:", y_train.shape)
        print("Test shape:", X_test.shape, "Labels:", y_test.shape)

        class_names = [str(i) for i in range(4)]

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
    # No datasets chosen / enviromental variable is wrong

    else:

        print("Invalid Dataset selected")
        exit(0)
        
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X).tolist()
    
    return X_scaled, y, None, None, class_names


# ==================================================================================================

def main():

    # X_scaled, y, class_names = load_dataset()
    
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

                    if(DATASET in ("cifar3", "mnist", "mnist4")):
                        features = X_train[index].ravel().astype(np.float32)
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

                        if(DATASET in ("cifar3", "mnist", "mnist4")):
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

                        if(DATASET in ("cifar3", "mnist", "mnist4")):
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

    
if __name__ == "__main__":
    main()
                              
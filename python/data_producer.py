
import json, time, random
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"      # Logging Level: 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from kafka import KafkaProducer
import argparse
import numpy as np
import random
import pandas as pd
import struct
from array import array

# ========================================================================================
# Env + Args =============================================================================

from dotenv import load_dotenv
loaded = load_dotenv("../java/.env")
print("Dotenv loaded:", loaded)

DATASET = os.getenv("DATASET")

PREDICTION_INPUT_TOPIC = os.getenv("PREDICTION_INPUT_TOPIC")
NUMBER_OF_DATA_REPEATS = int(os.getenv("NUMBER_OF_DATA_REPEATS"))
NUMBER_OF_DATA_REPEATS_TEST = int(os.getenv("NUMBER_OF_DATA_REPEATS_TEST"))

BATCH_FLUSH = int(os.getenv("BATCH_FLUSH"))

if(DATASET != "iris" and DATASET != "wine" and DATASET != "mnist"):
    NUMBER_OF_DATA_REPEATS = 1
    NUMBER_OF_DATA_REPEATS_TEST = 1

print(f"NUMBER_OF_DATA_REPEATS: {NUMBER_OF_DATA_REPEATS}")
print(f"NUMBER_OF_DATA_REPEATS_TEST: {NUMBER_OF_DATA_REPEATS_TEST}")

parser = argparse.ArgumentParser()
parser.add_argument('--streaming', action='store_true')
parser.add_argument('--all', action='store_true') # make this a flag argument
parser.add_argument('--pred', action='store_true')
parser.add_argument('--eval', action='store_true')
args = parser.parse_args()

if args.pred:
    print("Outputting to the prediction topic")
    INPUT_TOPIC = PREDICTION_INPUT_TOPIC
else:
    # INPUT_TOPIC = DATA_TOPIC
    INPUT_TOPIC = DATASET + "-input"
    TEST_TOPIC = DATASET + "-test"

print(f"Running this on input topic: {INPUT_TOPIC}")

# ========================================================================================
# Kafka Producer =========================================================================

def serialize_data_message(sample_index: int, features: np.ndarray, label: int) -> bytes:
    features = np.asarray(features, dtype=np.float32)
    n = int(features.size)

    header = struct.pack(">iii", int(sample_index), int(label), n)

    # Convert to big-endian float32 without copying when possible
    be = features.astype(">f4", copy=False)

    return header + be.tobytes()

# producer = KafkaProducer(
#     bootstrap_servers = "localhost:9092",
#     value_serializer = lambda v: json.dumps(v).encode("utf-8") # convert json int bytes before sending
# )

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

def evaluate_dataset(X_train, y_train, X_test, y_test):
    print("Train shape:", X_train.shape, "y range:", (int(y_train.min()), int(y_train.max())),
        "counts:", np.bincount(y_train, minlength=7))     # how many samples you have for each class
    print("Test  shape:", X_test.shape,  "y range:", (int(y_test.min()), int(y_test.max())),
        "counts:", np.bincount(y_test, minlength=7))



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
        (X_train, y_train), _ = mnist.load_data()
        X = X_train[:150].reshape(-1, 28 * 28).astype("float32")  # [60000, 784], by default mnist has 60000 samples
        y = y_train
        class_names = [str(i) for i in range(10)]           # "0".."9" each is one different number

    # ==================================================================================================

    elif DATASET == "susy":
        
        # data = np.loadtxt("../data/SUSY.csv", delimiter=",", max_rows=80000) # 80000 - 100000
                                                                             # (5000000, 19), the 19th is the label
        data = np.loadtxt("../data/SUSY.csv", delimiter=",", max_rows=80000, skiprows=80000)
        y_all = data[:, 0].astype(int)
        X_all = data[:, 1:].astype(np.float32)

        # ===== Train/Test Split =====
        train_size = 60000
        
        X_all, y_all = shuffle(X_all, y_all)

        X_train = X_all[:train_size]          # training features
        y_train = y_all[:train_size]          # training labels

        X_test = X_all[train_size:]     # test features (5000)
        y_test = y_all[train_size:]     # test labels  (5000)

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
    
    elif DATASET == "pendigits":
        NUMBER_OF_DATA_REPEATS = NUMBER_OF_DATA_REPEATS_TEST = 5
        base_path="../data"
        
        train_path = os.path.join(base_path, "pendigits.tra")
        test_path  = os.path.join(base_path, "pendigits.tes")

        print(f"Loading from: {base_path}")

        train = np.loadtxt(train_path, delimiter=",", dtype=np.float32)
        test  = np.loadtxt(test_path,  delimiter=",", dtype=np.float32)

        X_train = train[:, :-1].astype(np.float32)     # (n, 16)
        y_train = train[:, -1].astype(np.int64)        # (n,)
        X_test  = test[:, :-1].astype(np.float32)
        y_test  = test[:, -1].astype(np.int64)

        class_names = [str(i) for i in range(10)]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        evaluate_dataset(X_train, y_train, X_test, y_test)

        return X_train.tolist(), y_train, X_test.tolist(), y_test, class_names

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
    
    index = 0
    data_repeats = 0
    print(class_names)
    
    if args.eval:
        exit(0)
        
    # =================================================================================================================

    if args.all:   
         
        # Load to Training Topic ==================================================================
        if(1 == 1):
            
            while data_repeats < NUMBER_OF_DATA_REPEATS:
                
                for index in range(len(X_train)):
                    features = X_train[index]
                    label = int(y_train[index])
                    try:
                        label_name = class_names[int(label)]
                    except Exception as e:
                        print("ERROR while indexing class_names")
                        print("label value:", label)
                        print("class_names length:", len(class_names))
                        raise

                    msg = {
                        "sample_index": index,
                        "features": features,
                        "label": label
                    }

                    producer.send(INPUT_TOPIC, value=msg)
                    
                    if index % BATCH_FLUSH == 0:
                        producer.flush()

                producer.flush()

                data_repeats += 1
            
            print(f"Loaded entire {DATASET} dataset in {INPUT_TOPIC}")
        
        # Load to Test Topic =====================================================================
        
        if(1 == 1):
                
            data_repeats = 0
            
            if (X_test.any()) or (y_test.any()):     
                while data_repeats < NUMBER_OF_DATA_REPEATS_TEST:
                    
                    for index in range(len(X_test)):
                        features = X_test[index]
                        label = int(y_test[index])
                        label_name = class_names[label]

                        msg = {
                            "sample_index": index,
                            "features": features,
                            "label": label
                        }

                        producer.send(TEST_TOPIC, value=msg)
                        producer.flush() 
                    
                    data_repeats += 1
                                
                print(f"Loaded entire {DATASET} dataset in {TEST_TOPIC}")
                
            else:   # If there are no explicit training samples then load the training samples in the test topic
                
                while data_repeats < NUMBER_OF_DATA_REPEATS_TEST:
                    
                    for index in range(len(X_train)):
                        features = X_train[index]
                        label = int(y_train[index])
                        label_name = class_names[label]

                        msg = {
                            "sample_index": index,
                            "features": features,
                            "label": label
                        }

                        producer.send(TEST_TOPIC, value=msg)
                        producer.flush() 
                    
                    data_repeats += 1
                                
                print(f"Loaded entire {DATASET} dataset in {TEST_TOPIC}")
    
    # =================================================================================================================
      
    elif args.streaming:
        
        while True:
            index = random.randrange(len(X_train))      # we need random samples (if in order, they would belong to the same class)
            features = X_train[index]
            label = int(y_train[index])
            label_name = class_names[label]

            msg = { 
                "sample_index" : index,
                "features": features,
                "label": label,
                "label_name": label_name
            }

            producer.send(INPUT_TOPIC, value=msg) # Kafka Producer doesnt send immidiately, it buffers messages into a queue and sends them in batches
            producer.flush()                # This sends everything that been buffered

            print(f"sent: {msg}")
            index = index + 1
            
            time.sleep(1)

    else:
        print("Arg missing")

    
if __name__ == "__main__":
    main()
                              
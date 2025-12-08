
import json, time, random
import os
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from kafka import KafkaProducer
import argparse
import numpy as np
import random

from dotenv import load_dotenv
loaded = load_dotenv("../java/.env")
print("Dotenv loaded:", loaded)

DATASET = os.getenv("DATASET")
DATA_TOPIC = os.getenv("DATA_TOPIC")
PREDICTION_INPUT_TOPIC = os.getenv("PREDICTION_INPUT_TOPIC")
NUMBER_OF_DATA_REPEATS = int(os.getenv("NUMBER_OF_DATA_REPEATS"))

parser = argparse.ArgumentParser()
parser.add_argument('--streaming', action='store_true')
parser.add_argument('--all', action='store_true') # make this a flag argument
parser.add_argument('--pred', action='store_true')
args = parser.parse_args()

if args.pred:
    INPUT_TOPIC = PREDICTION_INPUT_TOPIC
else:
    # INPUT_TOPIC = DATA_TOPIC
    INPUT_TOPIC = DATASET + "-input"
    
    
producer = KafkaProducer(
    bootstrap_servers = "localhost:9092",
    value_serializer = lambda v: json.dumps(v).encode("utf-8") # convert json int bytes before sending
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


def load_susy_sample1(path, sample_size=60000):     # choose specific labels equally
    per_class = sample_size // 2
    zeros = []
    ones = []
    seen0 = 0
    seen1 = 0

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts or len(parts) < 2:
                continue

            *feat_strs, label_str = parts
            label = int(float(label_str))  # SUSY labels are 0/1 but may be "0.0"

            features = [float(v) for v in feat_strs]

            if label == 0:
                seen0 += 1
                if len(zeros) < per_class:
                    zeros.append((features, label))
                else:
                    j = random.randrange(seen0)
                    if j < per_class:
                        zeros[j] = (features, label)
            else:
                seen1 += 1
                if len(ones) < per_class:
                    ones.append((features, label))
                else:
                    j = random.randrange(seen1)
                    if j < per_class:
                        ones[j] = (features, label)

    # combine and shuffle
    data = zeros + ones
    random.shuffle(data)

    X = np.array([row[0] for row in data], dtype=np.float32)
    y = np.array([row[1] for row in data], dtype=np.int64)

    return X, y


# ========================================================================================

def load_dataset():
    X = y = class_names = None
    
    if DATASET == "iris":
        iris = load_iris()
        X = iris.data
        y = iris.target
        class_names = iris.target_names.tolist()

    elif DATASET == "wine":   # <-- remove the space in "wine -input"
        wine = load_wine()
        X = wine.data              # shape (178, 13)
        y = wine.target            # 0,1,2
        class_names = wine.target_names.tolist()
    
    elif DATASET == "mnist":
        (X_train, y_train), _ = mnist.load_data()
        X = X_train[:150].reshape(-1, 28 * 28).astype("float32")  # [60000, 784], by default mnist has 60000 samples
        y = y_train
        class_names = [str(i) for i in range(10)]           # "0".."9" each is one different number
    
    # elif DATASET == "susy":
    #     data = np.loadtxt("../data/SUSY.csv", delimiter=",", max_rows=60000)    # (5000000, 19), the 19th is the label
    #     X = data[:, :-1]                 # all columns except last are the features
    #     y = data[:, -1].astype(int)      # last column is the label label
    #     class_names = [str(i) for i in sorted(set(y))]

    elif DATASET == "susy":
        data = np.loadtxt("../data/SUSY.csv", delimiter=",", max_rows=60000)  # (5000000, 19), the 19th is the label
        y = data[:, 0].astype(int)            # first column = label (0/1)
        X = data[:, 1:].astype(np.float32)    # remaining 18 columns = features
        class_names = [str(i) for i in sorted(set(y))]
     
    # elif DATASET == "susy":
    #     X, y = load_susy_sample1("../data/SUSY.csv", sample_size=60000) # (5000000, 19)
    #     class_names = [str(i) for i in sorted(set(y))]
            
    else:
        print("Invalid Dataset selected")
        exit(0)
        
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X).tolist()
    
    return X_scaled, y, class_names


def main():

    X_scaled, y, class_names = load_dataset()

    index = 0
    data_repeats = 0

    if args.all:
        
        while data_repeats < NUMBER_OF_DATA_REPEATS:
            
            for index in range(len(X_scaled)):
                features = X_scaled[index]
                label = int(y[index])
                label_name = class_names[label]

                msg = {
                    "sample_index": index,
                    "features": features,
                    "label": label
                }

                producer.send(INPUT_TOPIC, value=msg)
                producer.flush() 
            
            data_repeats += 1
            
        print(f"Loaded entire {DATASET} dataset in {INPUT_TOPIC}")
        
    elif args.streaming:
         
        while True:
            index = random.randrange(len(X_scaled)) # we need random samples (if in order, they would belong to the same class)
            features = X_scaled[index]
            label = int(y[index])
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

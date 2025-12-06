
import json, time, random
import os
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from kafka import KafkaProducer
import argparse

from dotenv import load_dotenv
loaded = load_dotenv("../java/.env")
print("Dotenv loaded:", loaded)

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
    INPUT_TOPIC = DATA_TOPIC

producer = KafkaProducer(
    bootstrap_servers = "localhost:9092",
    value_serializer = lambda v: json.dumps(v).encode("utf-8") # convert json int bytes before sending
)

def load_dataset():
    X = y = class_names = None
    
    if DATA_TOPIC == "iris-input":
        iris = load_iris()
        X = iris.data
        y = iris.target
        class_names = iris.target_names.tolist()

    elif DATA_TOPIC == "wine-input":   # <-- remove the space in "wine -input"
        wine = load_wine()
        X = wine.data              # shape (178, 13)
        y = wine.target            # 0,1,2
        class_names = wine.target_names.tolist()

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
            
        print(f"Loaded entire iris dataset in iris-input topic")
        
    elif args.streaming: 
        while True:
            index = random.randrange(len(X_scaled)) # we need random samples (if in order, they would belong to the same class)
            features = X_scaled[index]
            label = int(y[index])
            label_name = class_names[label]

            msg = { 
                "sample_index" : index,
                "features": features,
                "label": label
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

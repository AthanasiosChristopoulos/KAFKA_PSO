
import json, time, random
import os
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from kafka import KafkaProducer
import argparse

from dotenv import load_dotenv
load_dotenv()
DATA_TOPIC = os.getenv("DATA_TOPIC")
NUMBER_OF_DATA_REPEATS = int(os.getenv("NUMBER_OF_DATA_REPEATS"))

parser = argparse.ArgumentParser()
parser.add_argument('--all', action='store_true') # make this a flag argument
args = parser.parse_args()

producer = KafkaProducer(
    bootstrap_servers = "localhost:9092",
    value_serializer = lambda v: json.dumps(v).encode("utf-8") # convert json int bytes before sending
)

iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names.tolist()

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X).tolist()
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

            producer.send(DATA_TOPIC, value=msg)
            producer.flush() 
        
        data_repeats += 1
        
    print(f"Loaded entire iris dataset in iris-input topic")
    
else: 
    while True:
        index = random.randrange(len(X)) # we need random samples (if in order, they would belong to the same class)
        features = X_scaled[index]
        label = int(y[index])
        label_name = class_names[label]

        msg = { 
            "sample_index" : index,
            "features": features,
            "label": label
        }

        producer.send(DATA_TOPIC, value=msg) # Kafka Producer doesnt send immidiately, it buffers messages into a queue and sends them in batches
        producer.flush()                # This sends everything that been buffered

        print(f"sent: {msg}")
        index = index + 1
        
        time.sleep(1)

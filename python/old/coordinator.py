# file: coordinator.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse, json, logging, time
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
import tensorflow as tf
from tensorflow.keras.layers import Dense

from dotenv import load_dotenv
load_dotenv()
N_WORKERS = int(os.getenv("N_WORKERS"))
DATA_TOPIC = os.getenv("DATA_TOPIC")
LOCAL_WEIGHTS_TOPIC = os.getenv("LOCAL_WEIGHTS_TOPIC")
GPEST_WEIGHTS_TOPIC = os.getenv("GPEST_WEIGHTS_TOPIC")
ENABLE_LOGGING = int(os.getenv("ENABLE_LOGGING"))

n_predictions = 0
n_correct_predictions = 0

pBest_dictionary = {}

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),  # Input Layer, 4 because we have 4 features
    tf.keras.layers.Dense(16, activation="relu"),  # Hidden Layer 1
    tf.keras.layers.Dense(16, activation="relu"),  # Hidden layer 2
    tf.keras.layers.Dense(3, activation="softmax")  # Output Layer  
])

# =======================================================================================================================================
# =======================================================================================================================================

data_consumer = KafkaConsumer(
    DATA_TOPIC,
    bootstrap_servers = ["localhost:9092"],
    group_id = "coordinator_eval",
    enable_auto_commit = False,
    auto_offset_reset = "earliest",
    value_deserializer = lambda b: json.loads(b.decode()),
    key_deserializer = lambda b: b.decode() if b else None,
)

consumer = KafkaConsumer(
    LOCAL_WEIGHTS_TOPIC,
    bootstrap_servers = ["localhost:9092"],
    group_id = "coordinator_group",
    enable_auto_commit = True,
    auto_offset_reset = "latest",
    value_deserializer = lambda b: json.loads(b.decode()),
    key_deserializer = lambda b: b.decode() if b else None,
)


producer = KafkaProducer(
    bootstrap_servers = ["localhost:9092"],
    value_serializer = lambda v: json.dumps(v).encode(),
    key_serializer = lambda k: k.encode(),
)

# =======================================================================================================================================
# =======================================================================================================================================

def sample_flat(flat, n=3):
    str_builder = "["
    count = 0
    for x in flat[:n]:
        str_builder += f"{round(x, 3)}"
        count += 1
        str_builder += ", "
    str_builder += " ... ]"
    return str_builder

# =======================================================================================================================================
# =======================================================================================================================================

def update_model(model, flat):
    idx = 0
    for layer in model.layers:
        if isinstance(layer, Dense):
            W, b = layer.get_weights()
            before_layer_neurons, current_layer_neurons = W.shape

            for j in range(current_layer_neurons):
                for i in range(before_layer_neurons):
                    W[i, j] = flat[idx];   # we follow convention: 
                        # the first element in the flat list should be the weight for the first edge of the first neuron of the first layer.
                    idx += 1
                    
                b[j] = flat[idx]; idx += 1
            layer.set_weights([W, b])
            
# =======================================================================================================================================
# =======================================================================================================================================

def start_from_the_beginning():
    
    for _ in range(10): 
        data_consumer.poll(timeout_ms=100)
        assignment = data_consumer.assignment()
        if assignment:
            data_consumer.seek_to_beginning(*assignment)
        
# =======================================================================================================================================
# =======================================================================================================================================

def call_predictions_batch(msgs): # Multiple samples at a time
    global n_predictions, n_correct_predictions

    if not msgs:
        return None

    objs = [m.value for m in msgs]
    X = np.asarray([o["features"] for o in objs])
    y = np.asarray([int(o["label"]) for o in objs])

    probs = model(X, training=False).numpy()
    preds = probs.argmax(axis=1)
    
    correct_mask = (preds == y)

    n = len(objs)
    n_correct = int(correct_mask.sum())
    n_predictions += n
    n_correct_predictions += n_correct

# =======================================================================================================================================
# =======================================================================================================================================

def main():
    global n_predictions, n_correct_predictions

    print("Coordinator started")
    
    os.makedirs("logs", exist_ok = True)
    log_path = "logs/coordinator.log"
    
    logging.basicConfig(filename = log_path, 
                        filemode = "w", 
                        level = logging.INFO,
                        format = "%(message)s"
    )
    
    if(ENABLE_LOGGING == 1):
        logging.basicConfig(
            filename=log_path,
            filemode="w",
            level=logging.INFO,
            format="%(message)s",
        )
    else:
        logging.basicConfig(level=logging.CRITICAL + 1) # Silence if logging disabled. This is needed for performance reasons
        
    
    logging.getLogger("kafka").setLevel(logging.CRITICAL + 1)
    logging.getLogger("kafka.producer").setLevel(logging.CRITICAL + 1)
    logging.getLogger("kafka.consumer").setLevel(logging.CRITICAL + 1)

    logging.info(f"I am the coordinator")

    global_round = 0
    
    start_from_the_beginning()
    
    try:
        while True:
            bufs = []
            worker_id_seen = set()
            
            while(len(bufs) < N_WORKERS):
                
                records = consumer.poll(timeout_ms=5000)   # Ask the broker for any new messages, wait up to 500 milliseconds, 
                                                            # and return whatever batch (‘pack’) of record arrives.
                        
                if not records:
                    assignment = consumer.assignment()
                    if assignment:
                        for tp in assignment:
                            pos = consumer.position(tp)
                            logging.info(f"Received no messages, in topic {LOCAL_WEIGHTS_TOPIC}, partitions: {tp.partition}, offset={pos}")
                    continue
                
                for _, msgs in records.items(): 
                    for msg in msgs:
                        json_values = msg.value 
                        worker_id = json_values.get("worker_id")
                        
                        if(worker_id in worker_id_seen):
                            continue
                        
                        weights = json_values.get("weights")
                            
                        reached_acc_goal = json_values.get("reached_acc_goal")
                        if(reached_acc_goal == 1):  # when the accuracy is good enough, then stop
                            
                            logging.info("Reached accuracy goal, ending training ...")
                            print("Coordinator: Reached accuracy goal, ending training ...")
                            while True: # Consumes rest of the messages, so that next time coordinator wont read messages with "reached_acc_goal" field
                                records = consumer.poll(timeout_ms=10000) 
                                if(not records):
                                    break
                            print("Coordinator: Quitting ...")
                            return
                        
                        pbest = json_values.get("pBest")
                        if pbest:
                            pBest_dictionary[worker_id] = pbest
                            logging.info(f"Worker {worker_id}: Updated pBest")
            
                        bufs.append(np.asarray(weights))
                        worker_id_seen.add(worker_id)
                        logging.info(f"Got weights from worker {worker_id}")
                        
                        if len(bufs) >= N_WORKERS:
                            break
                                            
                    if len(bufs) >= N_WORKERS:
                        break
            
            average = np.mean(bufs, axis=0).tolist()
            global_round += 1
            
            update_model(model, average)
            
            out = {
                "round": global_round,
                "global_weights": average,
                "neighbor_pBests" : list(pBest_dictionary.values()),
            }
            
            future = producer.send(GPEST_WEIGHTS_TOPIC, key=f"round-{global_round}", value=out)
            producer.flush()    
            meta = future.get(timeout=10)
            logging.info(f"====== Wrote to topic = {GPEST_WEIGHTS_TOPIC}, partition = {meta.partition}, offset = {meta.offset}, round = {global_round}")
            
            # ============================== Evaluate global model ==============================
            
            while True:    
                records = data_consumer.poll(timeout_ms = 500) 
                    
                read_messages = 0
                
                batch_msgs = []
                for _, msgs in records.items():
                    if msgs:
                        batch_msgs.extend(msgs)
                        read_messages += len(msgs)

                if batch_msgs:
                    call_predictions_batch(batch_msgs)
                    
                if(read_messages == 0):

                    if(n_predictions == 0):  # Guard against no data
                        logging.info(f"No predictions, no data")
                        assignment = data_consumer.assignment()
                        for tp in assignment:
                            pos = data_consumer.position(tp)
                            logging.info(f"Read from topic = {DATA_TOPIC}, partitions: {tp.partition}, offset={pos}")
                                
                        continue
                        
                    accuracy = n_correct_predictions / n_predictions
                    
                    eval_str = f"Evaluation done: n_samples = {n_predictions}, correct = {n_correct_predictions}, accuracy = {round(accuracy, 4)}"
                    print(eval_str)
                    logging.info(eval_str)

                    n_correct_predictions = 0
                    n_predictions = 0
                    
                    start_from_the_beginning()
                    break
            
    except KeyboardInterrupt:
        pass
    finally:
        try: data_consumer.close()
        except: pass
        try: consumer.close()
        except: pass
        try: producer.close()
        except: pass

if __name__ == "__main__":
    main()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import json
import argparse
import logging

from tensorflow.keras.layers import Dense
from kafka import KafkaConsumer, KafkaProducer
from kafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor

from dotenv import load_dotenv
load_dotenv()
N_WORKERS = int(os.getenv("N_WORKERS"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS"))
DATA_TOPIC = os.getenv("DATA_TOPIC")
PREDICTION_INPUT_TOPIC = os.getenv("PREDICTION_INPUT_TOPIC")
LOCAL_WEIGHTS_TOPIC = os.getenv("LOCAL_WEIGHTS_TOPIC")
GBEST_WEIGHTS_TOPIC = os.getenv("GBEST_WEIGHTS_TOPIC")
DESIRED_ACCURACY = float(os.getenv("DESIRED_ACCURACY"))
INERTIA = float(os.getenv("INERTIA"))
C_SOCIAL = float(os.getenv("C_SOCIAL"))
ENABLE_LOGGING = int(os.getenv("ENABLE_LOGGING"))

velocity_i = []
neighbor_pBests = []

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),  # Input Layer, 4 because we have 4 features (iris dataset)
    tf.keras.layers.Dense(16, activation="relu"),  # Hidden Layer 1
    tf.keras.layers.Dense(16, activation="relu"),  # Hidden layer 2
    tf.keras.layers.Dense(3, activation="softmax")  # Output Layer  
])

n_predictions = 0
n_correct_predictions = 0

accuracy_best = 0
pBest = []

consumer = KafkaConsumer(
    DATA_TOPIC,
    bootstrap_servers = ["localhost:9092"],
    group_id = "worker_group",
    enable_auto_commit = False,
    auto_offset_reset = "earliest",
    partition_assignment_strategy = [RoundRobinPartitionAssignor],  
    value_deserializer = lambda b: json.loads(b.decode()),
    key_deserializer=lambda v: v.decode("utf-8") if v else None,
)

producer = KafkaProducer(
    bootstrap_servers = ["localhost:9092"],
    value_serializer = lambda v: v.encode("utf-8"),
    key_serializer = lambda v: v.encode("utf-8") if v else None,
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

def model_to_flat_list(model):
    flat = []   # flat weight vector
    
    for layer in model.layers:    
        
        if isinstance(layer, Dense):   
            W, b = layer.get_weights()  # W shape: (before_layer_neurons, current_layer_neurons), b shape: (current_layer_neurons,)
            before_layer_neurons, current_layer_neurons = W.shape     
                # for a static input neuron, we have (# current_layer_neurons) == number of (output) edges of that neuron
                            
            for j in range(current_layer_neurons):          
                for i in range(before_layer_neurons):     
                    flat.append(float(W[i, j]))
                            # each W for a layer represents the edges that are between the before layer and the current layer
                flat.append(float(b[j]))    
    
    return flat

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

def randomize_model_weights(worker_id, sigma=0.1):
    logging.info(f"Model before initial randomization: {sample_flat(model_to_flat_list(model))}")
    
    np.random.seed(worker_id)  # deterministic, but different for each worker
    for layer in model.layers:
        if isinstance(layer, Dense):
            W, b = layer.get_weights()
            W += np.random.normal(0, sigma, size = W.shape)
            b += np.random.normal(0, sigma, size = b.shape)
            layer.set_weights([W, b])
        
    logging.info(f"Model after initial randomization: {sample_flat(model_to_flat_list(model))}")
            
       
def randomize_velocity(worker_id, sigma=0.01): 
    global velocity_i
    np.random.seed(worker_id)
    flat = np.array(model_to_flat_list(model), dtype=np.float32)
    velocity_i = np.random.normal(0.0, sigma, size=flat.shape).astype(np.float32)
           
              
def initialize_worker_variables(worker_id: int):
    randomize_model_weights(worker_id + 1337)
    randomize_velocity(worker_id + 4242)
    return model

# =======================================================================================================================================
# =======================================================================================================================================

def update_x():
    
    global velocity_i

    x_i = np.array(model_to_flat_list(model), dtype=np.float32)

    if velocity_i is None:
        velocity_i = np.zeros_like(x_i, dtype=np.float32)

    social_aggregate = np.zeros_like(x_i)

    for pBest_j in neighbor_pBests:     # if not yet defined (initialization this will simply not execute) and social_aggregate == 0

        # logging.info(f"pBests = {sample_flat(pBest_j)}")

        pBest_j = np.array(pBest_j, dtype=np.float32)
        
        p_i_j = np.random.rand(*x_i.shape).astype(np.float32)  # p_i_j random coefficient
        social_aggregate += p_i_j * (pBest_j - x_i)

    social_aggregate = social_aggregate * (C_SOCIAL / N_WORKERS)

    velocity_i_1 = INERTIA * velocity_i + social_aggregate
    x_i_1 = x_i + velocity_i_1

    update_model(model, x_i_1.tolist())
    
    logging.info(f"New model = {sample_flat(model_to_flat_list(model))}")
    
    velocity_i = velocity_i_1.copy()

# =======================================================================================================================================
# =======================================================================================================================================

def call_prediction_single(obj):
    
    global n_predictions, n_correct_predictions
    sample_index = int(obj["sample_index"])
    features = obj["features"]
    label = int(obj["label"])
    
    x = np.array(features).reshape(1, 4)
    probs = model(x, training=False).numpy()[0] # this is the inference step (without backpropagation)
                                                # [0] => predict one sample at a time (flattens array of tensors)
    pred = int(np.argmax(probs))
    correct = int(pred == label)
    
    n_predictions = n_predictions + 1
    
    if(correct == 1):
        n_correct_predictions = n_correct_predictions + 1
        
    out = {
        "sample_index" : sample_index,
        "pred_class_index": pred,
        "pred_confidence": float(probs[pred]),
        "label": label,
        "correct": correct,
    }
    
    return json.dumps(out)


def call_predictions_batch(msgs): # Multiple samples at a time
    global n_predictions, n_correct_predictions

    if not msgs:
        return None

    objs = [m.value for m in msgs]
    X = np.array([obj["features"] for obj in objs])
    y = np.array([int(obj["label"]) for obj in objs])

    probs = model(X, training=False).numpy()
    preds = probs.argmax(axis=1)
    
    correct_mask = (preds == y)

    n = len(objs)
    n_correct = int(correct_mask.sum())
    n_predictions += n
    n_correct_predictions += n_correct

    return

# =======================================================================================================================================
# =======================================================================================================================================

def start_from_the_beginning():
    
    for _ in range(10): 
        consumer.poll(timeout_ms=100)
        assignment = consumer.assignment()
        if assignment:
            consumer.seek_to_beginning(*assignment)
            return
        
# =======================================================================================================================================
# =======================================================================================================================================
   
def main():
    
    global accuracy_best, pBest, n_predictions, n_correct_predictions, neighbor_pBests
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0)
    args = parser.parse_args()

    print(f"Worker with ID: {args.id} started")
    
    start_from_the_beginning() 
    
    global_consumer = KafkaConsumer(
        GBEST_WEIGHTS_TOPIC,
        bootstrap_servers=["localhost:9092"],
        group_id=f"worker_global_{args.id}",
        enable_auto_commit=True,
        auto_offset_reset="latest",
        value_deserializer = lambda b: json.loads(b.decode()),
        key_deserializer=lambda v: v.decode("utf-8") if v else None,
    )

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/particle_{args.id}.log"

    if(ENABLE_LOGGING == 1):
        logging.basicConfig(
            filename=log_path,
            filemode="w",
            level=logging.INFO,
            format="%(message)s",
        )
    else:
        logging.basicConfig(level=logging.CRITICAL + 1) # Silence if logging disabled. This is needed for performance reasons
        
    
    logging.info(f"I am worker with id = {args.id}")

    logging.getLogger("kafka").setLevel(logging.WARNING)
    logging.getLogger("kafka.producer").setLevel(logging.WARNING)
    logging.getLogger("kafka.consumer").setLevel(logging.WARNING)
    
    initialize_worker_variables(args.id)

    try:
        epoch = 0
        worker_send_increment = 0
        updated_pBest = False
        
        while True:     # Loops through epochs
            records = consumer.poll(timeout_ms = 500) # poll consumer, wait if there is any update to read.
            read_messages = 0
            
            batch_msgs = []
            for _, msgs in records.items():
                if msgs:
                    batch_msgs.extend(msgs)
                    read_messages += len(msgs)

            if batch_msgs:
                call_predictions_batch(batch_msgs)

            if(read_messages == 0): # condition for i-th particle to report. This should execute only if there are no more records left
                
                epoch += 1
                                
                if(n_predictions == 0):  # Guard against no data
                    logging.info(f"No predictions, no data")
                    continue
                
                accuracy = n_correct_predictions / n_predictions
                
                if(accuracy > accuracy_best):
                    weights = model_to_flat_list(model)
                    accuracy_best = accuracy
                    pBest = weights 
                    updated_pBest = True
                    logging.info(f"Found pBest with accuracy = {round(accuracy, 4)}")
                    
                logging.info(f"Evaluation done: n_samples = {n_predictions}, correct = {n_correct_predictions}, accuracy = {round(accuracy, 4)}")

                n_correct_predictions = 0
                n_predictions = 0
                                
                start_from_the_beginning()
                update_x()
                
                # =========================================================================================================
                
                if(epoch >= MAX_EPOCHS):
                    weights = model_to_flat_list(model)
                    worker_send_increment += 1

                    payload = {
                        "worker_id": str(args.id),
                        "worker_send_increment" : worker_send_increment,
                        "weights": weights,
                    }
                    
                    if(updated_pBest):
                        payload["pBest"] = pBest
                        
                    if(accuracy_best > DESIRED_ACCURACY):
                        logging.info(f"=============== Accuracy Goal Achieved with acc = {accuracy_best} ===============")
                        print(f"=============== Accuracy Goal Achieved with acc = {accuracy_best} ===============")
                        
                        payload["reached_acc_goal"] = 1
                                      
                    future = producer.send(LOCAL_WEIGHTS_TOPIC, key=str(args.id), value=json.dumps(payload))
                    producer.flush()
                    meta = future.get(timeout=10)
                    logging.info(f"============ Wrote to topic = {LOCAL_WEIGHTS_TOPIC}, partition = {meta.partition}, offset = {meta.offset}")

                    records = global_consumer.poll(timeout_ms=5000, max_records=1) # nothing arriving with 5 second, means training has stopped
                            # KafkaConsumer.poll()  is a fetch request to the Kafka broker that may block up to the specified timeout 
                            # if no new records are immediately available => as soon as there’s data available
                            
                    msg = None
                    for _, msgs in records.items():
                        if msgs:
                            msg = msgs[0]
                            break

                    assignment = global_consumer.assignment()
                    for tp in assignment:
                        pos = global_consumer.position(tp)
                        logging.info(f"Read from topic = {GBEST_WEIGHTS_TOPIC}, partitions: {tp.partition}, offset={pos}")
            
                    if not msg: 
                        logging.info(f"Stopped Training") # After the 5000 wait and coordinator hasnt answered, it will stop training ... 
                        break
                    
                    global_round = msg.value.get("round")
                    global_weights = msg.value.get("global_weights")
                    neighbor_pBests = msg.value.get("neighbor_pBests")
                    
                    # Sample:                    
                    logging.info(
                        f"Received for round {global_round} => global_weights = {sample_flat(global_weights)}, "
                        f"neighbor_pBests: length = {len(neighbor_pBests)}"
                    )
                    
                    logging.info(f"Received neighbor_pBests: length = {len(neighbor_pBests)}")
                    
                    
                    logging.info(f"Before central collection {sample_flat(model_to_flat_list(model))}")
                    update_model(model, global_weights)
                    logging.info(f"After central collection {sample_flat(model_to_flat_list(model))}")

                    updated_pBest = False
                    epoch = 0
            
    except KeyboardInterrupt:
        pass
    
    finally:
        try: consumer.close()
        except: pass
        try: global_consumer.close()
        except: pass
        try: producer.close()
        except: pass
        

if __name__ == "__main__":
    main()

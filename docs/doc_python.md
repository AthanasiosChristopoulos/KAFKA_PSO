pip install torch
pip install fastapi
pip install uvicorn
pip install kafka-python

```python

rho = np.random.rand(*xi.shape).astype(np.float32)
    xi.shape        # if the shape of xi is (3, 4), then it prints (3, 4)
    *(xi.shape)     # (3, 4) => 3, 4
    np.random.rand(3, 4)

x_i_1 = np.array([1.0, 2.0, 3.0])       # convert to numpy list
x_i_1.tolist()                          # [1.0, 2.0, 3.0], convert to tolist (the default pythony list) 

velocity_i = velocity_i_1           # then both variables point to the same array in memory.
velocity_i = velocity_i_1.copy()    # Using .copy() makes a separate array with the same values


```

```python
torch.save(model.state_dict(), "iris_model.pt") # write the model (weights and stuff to .pt), stores the state of the model to a file

```

```bash 

python3 -m venv .venv
source .venv/bin/activate  
pip install 'tensorflow==2.17.*' scikit-learn  # this means version 2.17 tensorflow and whatever the new scikit-learn is 

```

These are logits — unnormalized scores that represent how strongly the model leans toward each class before turning them into probabilities.
You convert logits → probabilities using the softmax function:
    For example:
    logits = [1.2, 0.3, -0.8] => softmax = [0.60, 0.27, 0.13] (true propabilities)

```python
float[] probs = softmax1xN(logits);
int pred = argmax(probs);

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),  # Input Layer, 4 because we have 4 features, 4 == Number of neurons in that layer. 
    tf.keras.layers.Dense(16, activation="relu"),   # Hidden Layer 1, 16 == Number of neurons in that layer. Since dense each of those 16 will 
                                                    # accept 4 inputs from the previous layer
    tf.keras.layers.Dense(16, activation="relu"),  # Hidden layer 2, 16 == Number of neurons in that layer. 
    tf.keras.layers.Dense(3)  # Output Layer, 3 == Number of neurons in that layer. 
            # 3 because we have 3 classes (these are logits, score of how strongly the model leans toward each class)
])

```


Kafka in Python

```python

    consumer = KafkaConsumer(
        "iris-input",
        bootstrap_servers=["localhost:9092"],
        group_id="iris-classification-app",  # processes that use this belong to the same group (with this group_id)
                        # knowing this, Kafka load balances partitions => assigns each partition to one of the consumers in the group
                        # consumers with the same group-id => parallel processing of kafka data
        enable_auto_commit=True,
        auto_offset_reset="latest",
        value_deserializer=lambda v: v.decode("utf-8"),
        key_deserializer=lambda v: v.decode("utf-8") if v else None,
    )
    
    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda v: v.encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8") if v else None,
    )

    msg_pack = consumer.poll(timeout_ms=3000)

    msg_pack are pairs of (tp, msgs), with tp being partitions and msgs being all the msg in that partition => {
        TopicPartition(topic='local-weights-topic-1', partition=0): [msg1, msg2, msg3],
        TopicPartition(topic='local-weights-topic-1', partition=1): [msg4, msg5],
    }

    for _, msgs in msg_pack.items():
        for m in msgs:
```
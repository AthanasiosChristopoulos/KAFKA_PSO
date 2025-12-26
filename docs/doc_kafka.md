
## Geometrically Monitored Particle Swarm Optimization for Data-Parallel Neural Training on Apache Kafka Streams
## Distributed Neural Network Training with Particle Swarm Optimization and Geometric Monitoring over Apache Kafka and Kafka Streams

```yml ================================================================================================================
KAFKA_NODE_ID: 1  # Single node, acts as both controller and broker 
KAFKA_PROCESS_ROLES: broker, controller
    # Broker (data plane): handles client traffic (produce/consume on topics).
    # Controller (control plane): stores cluster metadata and coordinates brokers (topic creation, partition leadership, reassignments, etc.).
        # You always need a controller to manage the Kafka Cluster

KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092, CONTROLLER://localhost:9093 # 0.0.0.0 means accept from any interface not just the ones from inside the docker container.
KAFKA_LISTENERS: PLAINTEXT://localhost:9092,CONTROLLER://localhost:9093 # Inside the container: localhost = the container itself (its loopback).
    # So Kafka binds to localhost:9092 which is INSIDE the container, it will reject any connections outside

KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092 # what the broker tells clients to use after they bootstrap.

KAFKA_NUM_PARTITIONS: 3  # This controls the default number of partitions that Kafka will give to any new topic

ports: "9092:9092" # tells Docker forward host port 9092 → container port 9092. “When someone connects to port 9092 on the host (like KafkaProducer(bootstrap_servers=["localhost:9092"]) ), take that traffic and forward it into port 9092 inside the container.”
    # Left side = host port.
    # Right side = container port.

```

## Kafka Explained:

**Offset:**
    the sequence number of a record within a partition. Each consumer maintains a position (the next offset it will read) per partition.

**Commit:**
    persisting your position so Kafka can restore it later. (Obviously the Consumer cant store the offset, since its a runtime program, it flushes memory)
     - this is done in the <code> __consumer_offsets </code> topic

     - enable_auto_commit = True: the client periodically commits the latest position in the background.
     - enable_auto_commit = False: you commit explicitly in the code (or never commit).

     - If there’s no commit yet, auto_offset_reset decides (earliest (from the beginning) vs latest).

    That said during execution the offset is still saved "keeps the offset in its local session state as long as it’s running"

**Partitioning:**
    - Kafka balances partitions, not individual messages, means it doesnt look at the record number of each partition 
    - Kafka Implements: Random / sticky partitioning 
    - Kafka balances each record in the partition using:
        - If you specify a key:
            - Record goes to the following partition
            - partition = hash(key) % num_partitions (this key comes from the consumers ID)

        - If you don’t specify a key:
            - Kafka uses a round-robin approach to distribute each record to each partition evenly
            - A producer sends to the broker, which randomly selects in which partition to put that record

    - Each partition in the topic is assigned to exactly one consumer within that group.
    
**Heartbeat:**
    - Every consumer that joins a consumer group (has a group_id) automatically starts a heartbeat thread.
    - That thread periodically sends a heartbeat request to the Kafka broker to tell it:
        - “I’m still alive and processing, don’t rebalance me away, kicking me out of the my consumer group”

**Broker Rebalance:**
    - Means that the broker is reassigning partitions to its consumers with the same group ID
    - The partitions in that broker remain static, what changes is which consumer reads what partitions within a group
    - Happens when:
        - A consumer with the same group_id joines or leaves 
        - Essentially this doesnt happen at normal operation when everything is stable but at the start and at the end of a run

**Logging Levels:**
    - DEBUG
    - INFO 
    - WARNING
    - ERROR
    - CRITICAL

    - setting levels: ignore all log messages below that level ... setLevel(Warning) , ignore DEBUG, and INFO
    - logging.getLogger("kafka").setLevel(logging.CRITICAL + 1) => silence everything completely

**poll**:
    - msg_pack = consumer.poll(timeout_ms=3000)
    - the Kafka consumer might already have messages stored locally in memory before you call poll().
        - Kafka sends data in batches of N bytes, but the consumer application might not require / use that many
        - buffered messages = messages already fetched from the broker but not yet returned to your application. 

```java ================================================================================================================
Properties properties = new Properties();
properties.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092"); // tells Kafka Streams app how to connect to the Kafka cluster.
    // how to bootstrap => you need the first broker to connect to (entry point to a network). This is where the Kafka Cluster lives.
    // Both the consumer and producer use the same broker connection to connect to the cluster.

StreamsBuilder builder = new StreamsBuilder();
KStream<String, String> in = builder.stream(INPUT_TOPIC, Consumed.with(Serdes.String(), Serdes.String())) 
    // you need to specify A) the input topic, B) the key / value types (How to turn the raw bytes from Kafka into Java objects)
    // in Kafka Streams, records are made out of keys and values
    // this chain just builds the graph. Calling these methods here does not process data and this code in it of it self doesnt run multiple 
    // times. However as data arrives, the operants in this topology will run multiple times (for each incoming record)

Topology topology = builder.build(); // the builder writes essentially the Processing topology. This is the finalizing step.

KafkaStreams streams = new KafkaStreams(topology, properties);
streams.start();

Runtime.getRuntime().addShutdownHook(new Thread(streams::close)); // this is the graceful shutdown hook, no data is lost + memory friendly

// Operations:

.map((key, value) -> ...) // can change the key
.mapValues(value -> ...)    // only for changing the value, its better for kafka   

KStream<String, String> out = in.mapValues(IrisStreamsApp::callPrediction); 
    // this IrisStreamsApp::callPrediction passes a method refernce object 

KStream<String, String> out = in.mapValues(value -> IrisStreamsApp.callPrediction(value));
    // Or this lambda function that calls that method

.filter((key, value) -> Long.parseLong(value) > 1000)

```

=================================================================================================================================
## Kafka Streams:

 - Use Case for Kafka Streams: 
    - a Kafka Streams app is usually a long-running service / app, you cant stop and restart constantly
        - its a constalty running stream
        - a topology (just processing rules) is static it cant be altered once you call .start() in the code 
    - you build a topology once, start it, and let it run indefinitely as events flow in.
    - as data streams in, it is processed immidiately. You cannot pause, dynamically rewire, or stop consuming data


 - Kafka Streams has two layers:
    - High-level DSL (StreamsBuilder, KStream)
    - Low-level Processor API (Processor) (when you just call .process() )

 - Kafka Streams DSL (Domain-Specific Language) => High Level Language on top of Kafka Streams Library
    - KStream, KTable, GlobalKTable

 - Processor API => costum logic (not really pure Kafka Streams logic, just Java, arbitery Java code)

=================================================================================================================================
## Tensorflow Java:

```bash

# print about the actuall model:
saved_model_cli show --dir iris_savedmodel --all

```
### Tensorflow Theory:

A matrix is a 2-D tensor.
A vector is a 1-D tensor.
A scalar is a 0-D tensor.

TensorFlow revolves around: Graph and Session
Computations are represented as graphs (mathematical operations) in TensorFlow
    => what computations exist (nodes) and how data flows between them (edges).
    => tensors are the data that are running through the graph

Α TensorFlow graph is a just schematic of the computation (no values, not running). A graph must be run inside a TensorFlow session.
Seesion: The runtime environment / executor that takes the graph + your data (tensors) and produces outputs (new tensors).

```java
Session sess = new Session(graph) // session depends on the graph
Tensor<Double> tensor = sess.runner().fetch("z")
  .feed("x", Tensor.<Double>create(3.0, Double.class))
  .feed("y", Tensor.<Double>create(6.0, Double.class))
  .run().get(0).expect(Double.class);
```

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs: // this is the metagraph with the tag serve 

SignatureDefs == named entry point. One model can have multiple signatures. The most usuall are:
    - signature_def['serve']
    - signature_def['serving_default']
inputs['keras_tensor'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 4) // this means the batch size is (X, 4). The number of samples you feed forward at once is X (and have always 4 features) 
                        => Allow any batch size
    name: serving_default_keras_tensor:0
The given SavedModel SignatureDef contains the following output(s):
outputs['output_0'] tensor_info:
    dtype: DT_FLOAT
    shape: (-1, 3)
    name: StatefulPartitionedCall_1:0

```java
    static { // Load Model. This runs once
        try {
            bundle = SavedModelBundle.load(SAVED_MODEL, SIGNATURE_TAG);  // loads the graph + variables that were exported from Python
                    // The SIGNATURE_TAG (often "serve" or "serving") tells TF which meta-graph inside the export to use

            // Discover first input/output from the serving signature   
            Map<String, SignatureDef> sigs = bundle.metaGraphDef().getSignatureDefMap();
            SignatureDef sig = sigs.getOrDefault("serving_default", sigs.values().stream().findFirst().orElseThrow());
                // SignatureDef describes which inputs and outputs the model expects when you run inference.

            inputName  = sig.getInputsMap().values().iterator().next().getName();  // input tensor name
            outputName = sig.getOutputsMap().values().iterator().next().getName(); // output tensor name
                // Without these names you can’t tell TF what data you’re feeding in or what you’re getting out.
                // These names are what you will later use in .feed(inputName, tensor) and .fetch(outputName) when running the session.
                
        } catch (Exception e) {
            throw new RuntimeException("TF init failed: " + e.getMessage(), e);
        }
    }

    TFloat32 input = TFloat32.tensorOf(Shape.of(1, 4))  // Normally Keras automatically converts input data under the hood into a tensor
                            // But in TensorFlow Java, there’s no abstraction, it expects explicit tensors for all inputs and outputs.
    for (int i = 0; i < 4; i++) {
        input.setFloat(features.get(i).getAsFloat(), 0, i);
    }

    var outList = bundle.session().runner()
        .feed(INPUT_TENSOR, input)
        .fetch(OUTPUT_TENSOR)
        .run();  // returns a list of output tensors (because models can have multiple outputs).

    TFloat32 probabilities_tensor = (TFloat32) outList.get(0) // our model has one output tensor (we arejust peeling the list layer)
    int n = (int) probabilities_tensor.shape().size(1);       // probabilities_tensor.shape() => (1,3) , then .size(1) => 3, so n outputs

```
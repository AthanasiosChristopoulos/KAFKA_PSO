
## Geometrically Monitored Particle Swarm Optimization for Data-Parallel Neural Training on Apache Kafka Streams
## Distributed Neural Network Training with Particle Swarm Optimization and Geometric Monitoring over Apache Kafka and Kafka Streams

```yml ===============================================================
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

## Kafka Explained: ===========================================================

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

## Kafka Retention Policy: ====================================================

There are TWO retention policies that Kafka supports:

 - Time-based retention:
    log.retention.hours=168   # keep up to 7 days long records 
    
 - Size-based retention:
    log.retention.bytes=3221225472   # ~3GB
    This means Kafka will keep at most ~3GB of log data per partition. When a partition grows beyond that size, Kafka will delete the oldest log segments until the size is back under the limit.

 - Kafka log storage works in a per log.segment.bytes=1073741824 basis. Logs are stored in segments of 1GB chucks. This means that retention will effectively work in ~1GB chunks, meaning a Kafka broker cant delete less than that (no fine-turning).

## Kafka Rebalancing: ====================================================

A rebalance is Kafka’s way of answering:
“Who in this group owns which partitions/tasks right now?”
    => rebalancing means assign partitions within that group
    => Whenever group membership changes (a member joins, leaves, crashes) => Kafka must compute a new assignment.
group.initial.rebalance.delay.ms explained:

Kafka consumers work in groups of an application. A group is how Kafka decides which consumer gets which partitions.
    => This assignment of partitions is called a rebalance 
Kafka waits briefly when a brand-new group, so that it wont have to do many rebalances afterwards

Without the delay, consumer A joins, Kafka assigns partitions.
A moment later consumer B joins, Kafka must revoke and reassign again.

That is a good tradeoff for long-running services, where saving a few startup seconds matters less than avoiding churn
Kafka is trading startup latency for assignment stability.

Why more threads matter:
In Kafka Streams, each stream thread is not “just a Java helper thread.” It is part of the processing membership/assignment machinery.
    => Its essentuially a whole new consumer, a Kafka stream thread is basically a consumer
    => Each StreamThread owns its own KafkaConsumer.
    
## Broker Parallelism / Multiple Brokers: ====================================================

In Kafka single node means single machine.
1 node = 1 server / 1 VM / 1 physical machine

🧠 3. Why multiple brokers on one machine cause random I/O

Kafka normally works like this:

👉 Each partition is written sequentially to a log file

BUT when you run many brokers on one disk:

each broker has its own logs

they all write at the same time

the disk has to jump between files

So instead of:
Sequential write to one log
you get:
Broker A writes → Broker B writes → Broker C writes → Broker A again...

🧠 4. Why multiple disks fixes it

If you instead have:

Broker A → Disk 1
Broker B → Disk 2
Broker C → Disk 3

then each disk gets sequential writes again

It mostly turns into multiple nodes logically, while still being one node physically (same disks, same NIC, same CPUs, same memory bus).

NIC = Network Interface Card network hardware: Broadcom 5720 Dual Port 1Gb On-Board LOM

1
191) time: 83.775, bestAccuracy: 0.772, bestLoss: 2.5142236, accuracy: 0.752, with nSamples: 500, nCorrect: 376 loss: 2.7200594, weights sample: [1.90353, 1.88747, -1.83364, ...], bestTrainingAccuracy: 0.91
[Worker1] Closed, because of idleness for 30021 ms
[Worker 1] Dist = 0.0733, Radius = 0.097, with velocity (magnitude): 10.297 => CONVERGED
[Worker 1] Average Elapsed Time per Batch: 31.724 ms, InActivePartitions 0
[Worker 1] Elapsed time: 83.908 seconds, exiting run()
[Worker0] Closed, because of idleness for 30025 ms
[Coordinator] Stop requested, closing streams
[Worker 0] Dist = 0.0741, Radius = 0.097, with velocity (magnitude): 10.832 => CONVERGED
Saved flat weights to models/global-model-flat.txt, with length=325
[Worker 0] Average Elapsed Time per Batch: 37.367 ms, InActivePartitions 0
[Worker 0] Elapsed time: 85.087 seconds, exiting run()
============= Training is over, ElapsedTime: 116.993 seconds =============
[Coordinator] Exiting run()
============== Coordinator stopped, stopping simulation ==============
[Coordinator] Final (Best) Results: Training Accuracy: 0.91, Test Accuracy:0.772
[Coordinator] Elapsed time: 116.081 seconds
achristopoulos@polytechnix:/mnt/nas_drive/achristopoulos/KAFKA_PSO_4$ 
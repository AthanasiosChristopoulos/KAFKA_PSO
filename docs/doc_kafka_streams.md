=====================================================================
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


## State Store: ==========================================================

There is local State Store (a state store is simply a local database) and remote State Store.
Both on Memory and on Disk (RocksDB)
A State Store, stores a key-value store.

## KTable: ============================================================================

A changelog is a stream of state changes over time.
A changelog stream interpreted as a table for each key, keep only the latest value:

```json
// KStream sees (full changelog):
("key"  , "table")
("gBest", "{json1}")
("gBest", "{json2}")
("gBest", "{json3}")
("gBest1", "{json4}")

// KTable sees only latest:
("gBest", "{json3}")
("gBest1", "{json4}")

```

```java

KTable<String, String> gBestTable = builder.table(  // table returns KTable<String, String> 
    GBEST_WEIGHTS_TOPIC,   // The Kafka topic GBEST_WEIGHTS_TOPIC as the source of truth
    Consumed.with(Serdes.String(), Serdes.String()),
        Materialized.<String, String, KeyValueStore<Bytes, byte[]>>as("gBestStore") // this is backed (disk) at /tmp/kafka-streams/APPLICATION_ID/...
        .withKeySerde(Serdes.String())  // For writing into the state store
        .withValueSerde(Serdes.String())
); 

dataStream
    .transform(
        () -> new BatchingTransformer(workerId, sharedState, BATCH_SIZE, N_BATCHES),
        "gBestStore"    // means this tranformer is dependent on that State Store (only then can you use that state store)
                        // Kafka Streams wires the store to that processor node (dataStream) in the topology graph.
    )
```

Explanation:
    - Materialized means: Store the result of this operation in a local state store. KTable is the logic. Materialized creates RocksDB
    - Materialized< Key, Value, StateStore > (the generics define those types) => Type of the StateStore how Kafka stores it internally
        - Materialized< String, String, KeyValueStore<Bytes, byte[]> > (key => Bytes, value => byte[])
    - as("gBestStore") => with this "key" you are going to find that State Store
    - KeyValueStore<Bytes, byte[]> => The read-only part is about the API you get in your processor, not low level data store internally.
        - Doesnt matter if its ReadOnlyKeyValueStore

```java

KeyValueStore<String, ValueAndTimestamp<String>> gBestStore = context.getStateStore("gBestStore"); // assign StateStore in a variable
gBestStore.get("gBest") // later you can get the keys from the gBestStore like this 

```

KTable stores Records as:
    Key: String
    Value: ValueAndTimestamp<String> // it needs the last update timestamp to know what the latest value is 

```java
// Caching:
KTable<String, String> gBestTable = builder.table(
    GBEST_WEIGHTS_TOPIC,
    Consumed.with(Serdes.String(), Serdes.String()),
    Materialized.<String, String, KeyValueStore<Bytes, byte[]>>as(stateStoreName)
        .withKeySerde(Serdes.String())
        .withValueSerde(Serdes.String())
        .withCachingDisabled() // disabling the caching into memory feature
);
```

By default, Kafka Streams uses a record cache in front of the state store:
 - Multiple updates for the same key are coalesced in memory.
 - They are written/flushed downstream:
     - When the cache fills
     - On commit intervals
 - Doesnt effect direct stateStore querries (they first query Memory not disk)

But when caching is disabled:
 - emit on each update, means the down code will execute

## ============================================================================
## KTable vs GlobalKTable:

A \texttt{GlobalKTable} differs from a standard \texttt{KTable} in that each Kafka Streams application instance maintains a fully replicated local copy of the table by consuming all partitions of the input topic. In contrast, a \texttt{KTable} is partitioned across instances with the same \texttt{application.id}.

Each Kafka Streams Instance must have the same application ID to be considered the same application

A KTable is partitioned across instances of the same Streams application.
    => keeps only the latest value per key (core functionality)
    => Each Instance stores only the keys for the partitions it owns (in the same application ID, every instance sees different partitions).
    => No single Instance sees the full KTable
    => If one topic has only 1 partition, then KTable ≈ GlobalKTable, since every Kafka Streams instance will get the same full KTable data

In a GlobalKTable (still performs the same core functionality, but):
    => Every Instance of the application gets all partitions of the topic and keeps a full copy of the table in a local state store.
        => An application (identified by a unique application.id) still keeps a local copy of the GlobalKTable 
        => Each application will build it own State Store / own GlobalKTable
        => the word “global” means “globally replicated from the topic,” not “one JVM-global singleton shared by everybody”

    => A GlobalKTable is not one shared in-memory table for the whole application. each Kafka Streams instance maintains its own local replica
    => Make every Streams instance consume all partitions of the INPUT_TOPIC into that store.
    => They don’t participate in Kafka Streams task scheduling
        => They dont create a task
        => They run a dedicated internal consumer thread (is separate from the stream threads)

## Why I use GlobalKTable:
 - To effectively perform parallelization using an additional thread. 
 - I cant do this with normal stream threads since state store is shared and topology cant be split into subtopologies
 - GlobalKTable creates its own independent subtopology, since the StateStore is instance independent, there is no dependency:
    => then Kafka Streams can split the topologies

## Processor API: ==========================================================================

```java

Transformer<InputKey, InputValue, OutputRecord> 
Transformer<String, String, KeyValue<String, String>> // KeyValue = Kafka Record

public class BatchingTransformer implements Transformer<String, String, KeyValue<String, String>> {...}
    // this is a custom Transformer pattern is build on top of Processor API (more limited than pure processor API)

builder.stream(...).transform(() -> new BatchingTransformer(...), "gBestStore") // this costum logic is called like this (this uses the DSL)


```

## ============================================================================
## Kafka Streams - Instances - Threads - Tasks:

- Hierarchy:
    - Machine
        - Instances (different application IDs)
            - Threads
                - Task (smallest unit of compute, spawns only if multiple partitions)
                
- Threads run tasks
- Each task = all processors (your KTable + your KStream + branches) for a given set of input partitions.

Each Kafka Streams Instance creates internally:
    - a Kafka client is a JVM process that uses Kafka client libraries => either a producer or a consumer
    - one or more consumers (for input topics)
    - one or more producers (for output / sink topics)
    - Kafka Streams creates many clients

Each task:
 - has its own instance of the processors (KTable internals, etc.),
 - has its own local state stores
 - is always processed by exactly one stream Thread,
 - runs records sequentially in the order of offsets for that partition (no parallelism inside a task).

- Topology: You build a topology with sources, processors, state stores, sinks.

Kafka Streams groups the topology into sub-topologies:
 - Sub-topology starts at one or more source topics and includes all downstream processors/stores/sinks that are connected to those sources.
 - Kafka Streams creates a new sub-topology only when pipelines are NOT connected by:
     - State stores
     - Joins
     - Repartition topics
     - Merges

## ============================================================================
## Higher Level Functions:

 - aggregate:

    ```java
    .aggregate( // Emits a new table update every time a new record arrives
        () -> null,
        (key, newValue, currentAggregate) -> { 
            ...
            return nextAggregate; // the nextAggregate will become the new aggregate for the next run (whatever you return from the aggregator)
        }
    )
    ```
## Metrics =======================================================================================

These are Kafka Streams metrics. streams.metrics() returns metrics for the entire KafkaStreams instance in that JVM:
    => This means all the internal consumers or producers Streams creates.
    => internal Kafka Producers get created when there is a .to(TOPIC_NAME) in the Kafka Streams code and there is writing to a topic
    => you cant measure time from the transformer. This isnt sending the object, this is just creating and returning it to the Kafka producer, who is actually going to send it.

 - request-latency-avg:
     - Average time for a Kafka Streams producer request to complete.
     - Includes: time waiting in client, network RTT, broker processing, and waiting for acknowledgements (depends on acks).
     - Interpretation: If this goes up, the broker/network is slower or you’re producing big batches or the broker is overloaded.

 - bufferpool-wait-time-total:
    - Cumulative time that producer threads spent blocked waiting for producer buffer memory (in case its already filled up).
    - This blocking happens when the producer has a bounded memory pool (buffer.memory). If it’s full (because broker/network can’t keep up), producer threads block waiting for free space. If this is >0 and growing fast, your pipeline is backpressured by producing => communication is the bottleneck
    - If this is 0, then there is no comminication bottleneck (there is no waiting for communication, just send it immidiately)
        => there is no communication pressure

 - outgoing-byte-rate:
     - Rate of bytes sent by the producer over the network (usually bytes/sec).
     - How “heavy” your producing is. Decreasing model size and filtering should reduce this.

 - record-send-rate:
     - Records / sec successfully sent by the producer.
     - record-send-rate = 0.3 records / sec

Kafka Streams Batches:
 - batch-size-avg => size of batch in bytes 
    - batch-size-max, is the biggest batch you ever sent. If batch-size-avg == batch-size-max then LINGERING == 0, batch size = 1
 - records-per-request-avg => number of records per batch in average
 - record-queue-time-avg [ms] => Depends on much linger.ms is, how long there is a wait specifically because of batch accumulations
    - is 0 if linger.ms is 0 as well

## ============================================================================
## What a Kafka produce request is / Kafka Streams Batching:

 - The Kafka producer batches records internally and periodically sends a network request to a broker leader.
 - That request may contain many records:
    - buffers (queues) records in memory, groups them into batches, and then sends produce requests containing those batches.

 - Kafka producer batching policy:
    Kafka Streams keeps appending records to the current batch for that partition until one of these conditions triggers send:
    - A) batch.size threshold reached (bytes)
    - B) linger.ms timeout expires => the batch can be sent even if not full
    - C) Pressure-driven behavior (buffer.memory & backpressure) => high congestion may send more quickly


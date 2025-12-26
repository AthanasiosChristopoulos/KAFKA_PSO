============================================================================
## State Store:
There is local State Store (a state store is simply a local database) and remote State Store.
Both on Memory and on Disk (RocksDB)
A State Store, stores a key-value store.

============================================================================
## KTable:

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
    GLOBAL_WEIGHTS_TOPIC,   // The Kafka topic GLOBAL_WEIGHTS_TOPIC as the source of truth
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
    GLOBAL_WEIGHTS_TOPIC,
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

============================================================================
## KTable vs GlobalKTable:

A KTable is partitioned across instances of the same Streams application.
    => Each Kafka Streams Instance must have the same application ID to be considered the same application
    => No single Instance sees the full KTable

In a GlobalKTable (still keeps only the latest value per key):
    => Every instance of the application gets all partitions of the topic, and keeps a full copy of the table in a local state store.
        => An application (identified by a unique application.id) still keeps a local copy of the GlobalKTable 
        => Each application will build it own State Store

    => Make every Streams instance consume all partitions of the INPUT_TOPIC into that store.
    => They don’t participate in Kafka Streams task scheduling
        => They dont create a task
        => They run a dedicated internal consumer thread (is separate from the stream threads.)

## Why i use GlobalKTable:
 - To effectively perform parallelization using an additional thread. 
 - I cant do this with normal stream threads since state store is shared and topology cant be split into subtopologies
 - GlobalKTable creates its own independent subtopology, since the StateStore is instance independent, there is no dependency:
    => then Kafka Streams can split the topologies


============================================================================
## Processor API:

```java

Transformer<InputKey, InputValue, OutputRecord> 
Transformer<String, String, KeyValue<String, String>> // KeyValue = Kafka Record

public class BatchingTransformer implements Transformer<String, String, KeyValue<String, String>> {...}
    // this is a custom Transformer pattern is build on top of Processor API (more limited than pure processor API)

builder.stream(...).transform(() -> new BatchingTransformer(...), "gBestStore") // this costum logic is called like this (this uses the DSL)


```

============================================================================
## Kafka Streams - Instances - Threads - Tasks:

 - Threads run tasks
 - Each task = all processors (your KTable + your KStream + branches) for a given set of input partitions.

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

============================================================================
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
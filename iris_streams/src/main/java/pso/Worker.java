
package pso;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.*;
import org.apache.kafka.streams.kstream.Consumed;
import org.apache.kafka.streams.kstream.Produced;
import org.apache.kafka.streams.kstream.Materialized;
import org.apache.kafka.streams.kstream.KTable;
import org.apache.kafka.streams.kstream.KStream;
import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.streams.state.Stores;
import org.apache.kafka.common.utils.Bytes;

import java.util.Properties;
import java.util.concurrent.CountDownLatch;

import utils.Config; // from the same project (in the utils package)

public class Worker implements Runnable {

    private final int workerId;
    
    private final int BATCH_SIZE;
    private final int N_BATCHES;
    private final String DATA_TOPIC;
    private final String LOCAL_WEIGHTS_TOPIC;
    private final String GLOBAL_WEIGHTS_TOPIC;
    private final String RUN_ID;

    public Worker(int workerId) {

        this.workerId = workerId;

        Config cfg = Config.get();
        this.BATCH_SIZE = cfg.BATCH_SIZE;
        this.N_BATCHES = cfg.N_BATCHES;         
        this.DATA_TOPIC = cfg.DATA_TOPIC;
        this.LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
        this.GLOBAL_WEIGHTS_TOPIC = cfg.GLOBAL_WEIGHTS_TOPIC;
        this.RUN_ID = cfg.RUN_ID;                   

    }

    @Override
    public void run() {

        System.out.println("[Worker " + workerId + "] with RUN_ID = " + RUN_ID);

        WorkerSharedState sharedState = new WorkerSharedState(workerId);

        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-worker-" + workerId + "_" + RUN_ID);
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092"); // for now localhost, but this is the URL of the Kafka cluster
        props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1");
        props.put(org.apache.kafka.clients.consumer.ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");

        StreamsBuilder builder = new StreamsBuilder();

        // KTable over GLOBAL_WEIGHTS_TOPIC, materialized as "gBestStore"
        KTable<String, String> gBestTable = builder.table(
            GLOBAL_WEIGHTS_TOPIC,
            Consumed.with(Serdes.String(), Serdes.String()),
            Materialized.<String, String, KeyValueStore<Bytes, byte[]>>as("gBestStore")
                .withKeySerde(Serdes.String())
                .withValueSerde(Serdes.String())
        );

        // Print: Optional to DEBUG
        // gBestTable
        //     .toStream()
        //     .peek((k, v) -> System.out.println(
        //         "[DEBUG GLOBAL] key = " + k + " value = " + v
        //     ));

        // 2) DATA stream → BatchingTransformer (which will read gBestStore)

        KStream<String, String> dataStream = builder.stream(
            DATA_TOPIC,
            Consumed.with(Serdes.String(), Serdes.String())
        );

        dataStream
            .transform(() -> new BatchingTransformer(workerId, sharedState, BATCH_SIZE, N_BATCHES),"gBestStore" )
                                                                //  wire the state store to the BatchingTransformer
            .filter((k, v) -> v != null)
            .to(LOCAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), Serdes.String()));

        System.out.println("[Worker " + workerId + " i am here2]");

        Topology topology = builder.build();
        System.out.println("[Worker " + workerId + "] Topology:");
        System.out.println(topology.describe());

        KafkaStreams streams = new KafkaStreams(topology, props);

        streams.setUncaughtExceptionHandler((Thread t, Throwable e) -> {
            System.out.println("[Worker " + workerId + "] Uncaught exception in thread " + t.getName());
            e.printStackTrace();
        });
        System.out.println("[Worker " + workerId + " i am here3]");

        CountDownLatch latch = new CountDownLatch(1);

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[Worker " + workerId + "] Shutting down KafkaStreams");
            streams.close();
            latch.countDown();
        }));

        try {
            streams.start();
            System.out.println("[Worker " + workerId + "] KafkaStreams started.");
            latch.await();
        } catch (Throwable e) {
            System.out.println("[Worker " + workerId + "] Error in KafkaStreams: " + e.getMessage());
            streams.close();
            latch.countDown();
        }

        System.out.println("[Worker " + workerId + "] Exiting run()");
    }
}


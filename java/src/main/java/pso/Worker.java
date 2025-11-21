
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

import utils.*;

public class Worker implements Runnable {

    private final int workerId;
    
    private final int BATCH_SIZE;
    private final int N_BATCHES;
    private final String DATA_TOPIC;
    private final String PBEST_WEIGHTS_TOPIC;
    private final String LOCAL_WEIGHTS_TOPIC;
    private final String GLOBAL_WEIGHTS_TOPIC;
    private final String RUN_ID;
    private final String FULLY_INFORMED;

    private String stateStoreName;
    private String keyName;

    private final CustomLogger logger;

    public Worker(int workerId) {

        this.workerId = workerId;

        Config cfg = Config.get();
        this.BATCH_SIZE = cfg.BATCH_SIZE;
        this.N_BATCHES = cfg.N_BATCHES;         
        this.DATA_TOPIC = cfg.DATA_TOPIC;
        this.PBEST_WEIGHTS_TOPIC = cfg.PBEST_WEIGHTS_TOPIC;
        this.LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
        this.GLOBAL_WEIGHTS_TOPIC = cfg.GLOBAL_WEIGHTS_TOPIC;
        this.RUN_ID = cfg.RUN_ID;   
        this.FULLY_INFORMED = cfg.FULLY_INFORMED;

        this.logger = CustomLogger.getWorkerInstance(workerId);                

        if("true".equals(FULLY_INFORMED)) {
            stateStoreName = "pBestStore";
            keyName = "pBest" + workerId;
        } else {
            stateStoreName = "gBestStore";
            keyName = "gBest";
        }
    }

    @Override
    public void run() {

        System.out.println("[Worker " + workerId + "] with RUN_ID = " + RUN_ID);

        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-worker-" + workerId + "_" + RUN_ID);
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092"); // for now localhost, but this is the URL of the Kafka cluster
        props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1");
        props.put(org.apache.kafka.clients.consumer.ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");

        StreamsBuilder builder = new StreamsBuilder();

        // KTable over WEIGHTS_TOPIC, materialized as "stateStoreName" ====================================
        if("true".equals(FULLY_INFORMED)) {
            KTable<String, String> gBestTable = builder.table(
                PBEST_WEIGHTS_TOPIC,
                Consumed.with(Serdes.String(), Serdes.String()),
                Materialized.<String, String, KeyValueStore<Bytes, byte[]>>as(stateStoreName)
                    .withKeySerde(Serdes.String())
                    .withValueSerde(Serdes.String())
                    .withCachingDisabled()
            );
            
            gBestTable
                .toStream()
                .peek((k, json) -> {
                    logger.log("Received New gBest JSON: " + json);
                    // System.out.println("[Coordinator] New gBest JSON: " + json);
                });

        } else {
            KTable<String, String> gBestTable = builder.table(
                GLOBAL_WEIGHTS_TOPIC,
                Consumed.with(Serdes.String(), Serdes.String()),
                Materialized.<String, String, KeyValueStore<Bytes, byte[]>>as(stateStoreName)
                    .withKeySerde(Serdes.String())
                    .withValueSerde(Serdes.String())
                    // .withCachingDisabled()
            );
        }

        // =====================================================================================================

        KStream<String, String> dataStream = builder.stream(
            DATA_TOPIC,
            Consumed.with(Serdes.String(), Serdes.String()))
            .transform(() -> new WorkerTransformer(workerId), stateStoreName)
                                                                //  wire the state store to the WorkerTransformer
            .filter((k, v) -> v != null);

        KStream<String, String>[] branches = dataStream.branch(
            (key, value) -> keyName.equals(key),   // branch[0]: pBest/gBest updates
            (key, value) -> true                   // branch[1]: all others (weights)
        );

        branches[0].to(PBEST_WEIGHTS_TOPIC, Produced.with(Serdes.String(), Serdes.String()));
        branches[1].to(LOCAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), Serdes.String()));

        // =====================================================================================================

        Topology topology;
        try {
            topology = builder.build();
        } catch (Exception e) {
            System.out.println("[Worker " + workerId + "] Didn't build topology:");
            e.printStackTrace();
            return;   
        }

        System.out.println("[Worker " + workerId + "] Topology:");
        // System.out.println(topology.describe());

        KafkaStreams streams = new KafkaStreams(topology, props);

        streams.setUncaughtExceptionHandler((Thread t, Throwable e) -> {
            System.out.println("[Worker " + workerId + "] Uncaught exception in thread " + t.getName());
            e.printStackTrace();
        });

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


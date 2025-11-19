
package pso;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.*;
import org.apache.kafka.streams.kstream.Consumed;
import org.apache.kafka.streams.kstream.Produced;
import org.apache.kafka.streams.KafkaStreams;

import java.util.Properties;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.logging.*;
import java.util.logging.Logger;

import java.util.concurrent.CountDownLatch;

public class Worker implements Runnable {

    private final int workerId;
    private final int BATCH_SIZE;
    private final int N_BATCHES;

    public Worker(int workerId) {
        this.workerId = workerId;

        this.BATCH_SIZE = Integer.parseInt(System.getenv().getOrDefault("BATCH_SIZE", "150"));
        this.N_BATCHES = Integer.parseInt(System.getenv().getOrDefault("N_BATCHES", "10"));
    }
    
    @Override
    public void run() {

        String RUN_ID = System.getenv().getOrDefault("RUN_ID", "111");
        String DATA_TOPIC = System.getenv().getOrDefault("DATA_TOPIC", "iris-input");
        String LOCAL_WEIGHTS_TOPIC = System.getenv().getOrDefault("LOCAL_WEIGHTS_TOPIC", "local-weights-topic");
        String GLOBAL_WEIGHTS_TOPIC = System.getenv().getOrDefault("GLOBAL_WEIGHTS_TOPIC", "global-weights-topic");

        System.out.println("[Worker " + workerId + "] Starting with BATCH_SIZE = " + BATCH_SIZE + " and with DATA_TOPIC = " + DATA_TOPIC);

        WorkerSharedState sharedState = new WorkerSharedState(workerId);

        // ====================================== DATA app (earliest) ===========================================

        Properties dataProps = new Properties();
        dataProps.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-worker-" + workerId + "_" + RUN_ID);
        // dataProps.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-worker-" + workerId + "_");
        dataProps.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        dataProps.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1");
        dataProps.put(org.apache.kafka.clients.consumer.ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");

        StreamsBuilder dataBuilder = new StreamsBuilder();


        // dataBuilder.stream(DATA_TOPIC, Consumed.with(Serdes.String(), Serdes.String()))
        //         .process(() -> new PredictionBatchProcessor(workerId, sharedState))
        //         .to(LOCAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), Serdes.String()));

        dataBuilder.stream(DATA_TOPIC, Consumed.with(Serdes.String(), Serdes.String()))
                    .transform(() -> new BatchingTransformer(workerId, sharedState, BATCH_SIZE, N_BATCHES))
                    // this outputs KeyValue<String, String> where value is the JSON
                    .filter((k, v) -> v != null) // only forward non-null results
                    .to(LOCAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), Serdes.String()));

        Topology dataTopology = dataBuilder.build();
        KafkaStreams dataStreams = new KafkaStreams(dataTopology, dataProps);

        dataStreams.setUncaughtExceptionHandler((Thread t, Throwable e) -> {
            System.out.println("[Worker " + workerId + "] Uncaught exception in thread " + t.getName());
            e.printStackTrace();
        });

        System.out.println("Worker " + workerId + " topology:");
        System.out.println(dataTopology.describe());


        // ====================================== GLOBAL app (latest) ===========================================
        Properties globalProps = new Properties();
        globalProps.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-worker-" + workerId + "-global_1");
        globalProps.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        globalProps.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1");
        globalProps.put(org.apache.kafka.clients.consumer.ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "latest");

        StreamsBuilder globalBuilder = new StreamsBuilder();
        globalBuilder.stream(GLOBAL_WEIGHTS_TOPIC, Consumed.with(Serdes.String(), Serdes.String()))
                    .process(() -> new PredictionBatchProcessor(workerId, sharedState));
        
        Topology globalTopology = globalBuilder.build();
        KafkaStreams globalStreams = new KafkaStreams(globalTopology, globalProps);

        globalStreams.setUncaughtExceptionHandler((Thread t, Throwable e) -> {
            System.out.println("[Worker " + workerId + "] Uncaught exception in thread " + t.getName());
            e.printStackTrace();
        });

        System.out.println(globalTopology.describe());

        CountDownLatch latch = new CountDownLatch(1); // this latch will block untit count goes from 1 to 0

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[Worker " + workerId + "] Shutting down KafkaStreams");
            dataStreams.close();
            globalStreams.close();
            latch.countDown();
        }));

        try {
            dataStreams.start();
            globalStreams.start();
            System.out.println("[Worker " + workerId + "] KafkaStreams started.");
            latch.await();   // block this worker thread until shutdown latch.countDown(); (one call)
        } catch (Throwable e) {
            System.out.println("[Worker " + workerId + "] Error in KafkaStreams: " + e.getMessage());
            dataStreams.close();
            globalStreams.close();
            latch.countDown();
        }

        System.out.println("[Worker " + workerId + "] Exiting run()");

        //==========================================================================================================

    }

}


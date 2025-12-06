
package pso;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.*;
import org.apache.kafka.streams.kstream.Consumed;
import org.apache.kafka.streams.kstream.Produced;
import org.apache.kafka.streams.kstream.Materialized;
import org.apache.kafka.streams.kstream.KTable;
import org.apache.kafka.streams.kstream.GlobalKTable;

import org.apache.kafka.streams.kstream.KStream;
import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.streams.state.Stores;
import org.apache.kafka.common.utils.Bytes;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.streams.processor.ThreadMetadata;
import org.apache.kafka.streams.processor.TaskMetadata;

import java.util.Properties;
import java.util.concurrent.CountDownLatch;

import utils.*;
import state.*;

public class Worker implements Runnable {

    private final int workerId;
    
    private final int BATCH_SIZE;
    private final int N_BATCHES;
    private final String DATA_TOPIC;
    private final String PBEST_WEIGHTS_TOPIC;
    private final String LOCAL_WEIGHTS_TOPIC;
    private final String GLOBAL_WEIGHTS_TOPIC;
    private final String RUN_ID;

    private final boolean FULLY_INFORMED;
    private final boolean DEBUG_KAFKA;

    private String stateStoreName;
    private String keyName;

    private final CustomLogger logger;
    private final CoordinatorControl control;

    public Worker(int workerId) {

        this.workerId = workerId;

        Config cfg = Config.getInstance();
        this.BATCH_SIZE = cfg.BATCH_SIZE;
        this.N_BATCHES = cfg.N_BATCHES;         
        this.DATA_TOPIC = cfg.DATA_TOPIC;
        this.PBEST_WEIGHTS_TOPIC = cfg.PBEST_WEIGHTS_TOPIC;
        this.LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
        this.GLOBAL_WEIGHTS_TOPIC = cfg.GLOBAL_WEIGHTS_TOPIC;
        this.RUN_ID = cfg.RUN_ID;   
        this.FULLY_INFORMED = cfg.FULLY_INFORMED;
        this.DEBUG_KAFKA = cfg.DEBUG_KAFKA;

        this.logger = CustomLogger.getWorkerInstance(workerId);                

        if(FULLY_INFORMED == true) {
            stateStoreName = "pBestStore";
            keyName = "pBest" + workerId;
        } else {
            stateStoreName = "gBestStore";
            keyName = "gBest";
        }

        this.control = CoordinatorControl.getInstance();
    }

    @Override
    public void run() {

        System.out.println("[Worker " + workerId + "] with RUN_ID = " + RUN_ID);

        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-worker-" + workerId + "_" + RUN_ID);
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092"); // for now localhost, but this is the URL of the Kafka cluster
        props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1");
        props.put(org.apache.kafka.clients.consumer.ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        // props.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 0);
        // props.put(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, 1);
        // props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "2");
        props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1"); // 2 is pointless. The Global table consumer thread takes care of task 0

        StreamsBuilder builder = new StreamsBuilder();

        // Task 0 (of Global Streams) ===============================================================================
        // input stream 4 and input stream 7

        if(FULLY_INFORMED == true) {

            GlobalKTable<String, String> gBestTable = builder.globalTable(
                PBEST_WEIGHTS_TOPIC,
                Consumed.with(Serdes.String(), Serdes.String()),
                Materialized.<String,   String, KeyValueStore<Bytes, byte[]>>as(stateStoreName)
                    .withKeySerde(Serdes.String())
                    .withValueSerde(Serdes.String())
            );

        } else {
            GlobalKTable<String, String> gBestTable = builder.globalTable(
                GLOBAL_WEIGHTS_TOPIC,
                Consumed.with(Serdes.String(), Serdes.String()),
                Materialized.<String, String, KeyValueStore<Bytes, byte[]>>as(stateStoreName)
                    .withKeySerde(Serdes.String())
                    .withValueSerde(Serdes.String())
            );
        }

        // Task 1 ================================================================================================
        // input stream 1 and output stream 2_1 and stream 2_2

        KStream<String, String> dataStream = builder.stream(
            DATA_TOPIC,
            Consumed.with(Serdes.String(), Serdes.String()))
            .transform(() -> new WorkerTransformer(workerId))
                                                                //  wire the state store to the WorkerTransformer
            .filter((k, v) -> v != null);

        KStream<String, String>[] branches = dataStream.branch(
            (key, value) -> keyName.equals(key),   // branch[0]: pBest/gBest updates
            (key, value) -> true                   // branch[1]: all others (weights)
        );

        branches[0].
            peek((k, v) -> {logger.log("Sending pBest: " + v);})
            .to(PBEST_WEIGHTS_TOPIC, Produced.with(Serdes.String(), Serdes.String()));
            
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

            System.out.println("[Worker " + workerId + "] started.");

            if(workerId == 0 && DEBUG_KAFKA == true) {
                System.out.println("[Worker " + workerId + "] Topology:\n" + topology.describe());

                try { 
                    Thread.sleep(1500); 
                } catch (InterruptedException ignored) {
                    System.out.println("Sleep failed");
                }

                for (ThreadMetadata tm : streams.localThreadsMetadata()) {
                    System.out.println("Thread: " + tm.threadName() + " state=" + tm.threadState());

                    for (TaskMetadata task : tm.activeTasks()) {
                        System.out.println("  ACTIVE Task: " + task.taskId()
                            + " partitions=" + task.topicPartitions());
                    }

                }
            }

            while (!control.isStopRequested()) {
                Thread.sleep(500); // poll every 500ms
            }
            System.out.println("[Worker " + workerId + " ] Stopping because desired accuracy was reached.");
            streams.close();

            latch.await();

        } catch (Throwable e) {
            System.out.println("[Worker " + workerId + "] Error in KafkaStreams: " + e.getMessage());
            streams.close();
            latch.countDown();
        }

        System.out.println("[Worker " + workerId + "] Exiting run()");
    }
}


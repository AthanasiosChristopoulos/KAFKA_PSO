package pso;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.common.serialization.Serde;
import org.apache.kafka.streams.*;
import org.apache.kafka.streams.kstream.Consumed;
import org.apache.kafka.streams.kstream.Produced;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.kstream.KStream;
import org.apache.kafka.streams.kstream.KTable;
import org.apache.kafka.streams.kstream.Materialized;
import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.common.utils.Bytes;
import org.apache.kafka.streams.kstream.Suppressed;
import org.apache.kafka.streams.processor.ThreadMetadata;
import org.apache.kafka.streams.processor.TaskMetadata;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.Properties;
import java.util.Map;
import java.util.HashMap;
import java.time.Duration;

import java.util.concurrent.CountDownLatch;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;


import utils.*; 
import state.*; 

public class Coordinator implements Runnable {

    public final String DATASET;
    private final String PBEST_WEIGHTS_TOPIC;
    private final String LOCAL_WEIGHTS_TOPIC;
    private final String GLOBAL_WEIGHTS_TOPIC;
    public final String PREDICTION_INPUT_TOPIC;
    public final String PREDICTION_OUTPUT_TOPIC;

    private final String RUN_ID;
    private final boolean FULLY_INFORMED;
    private final boolean DEBUG_KAFKA;

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private Map<String, Object> payload = new HashMap<>();

    private final CustomLogger logger;

    private final MultiLayerNetwork globalModel; // x_g , current model
    private final Stats globalStats;

    private final BatchPrediction predictor;

    public Coordinator() {

        Config cfg = Config.getInstance();
        this.PBEST_WEIGHTS_TOPIC = cfg.PBEST_WEIGHTS_TOPIC;
        this.LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
        this.GLOBAL_WEIGHTS_TOPIC = cfg.GLOBAL_WEIGHTS_TOPIC;
        this.PREDICTION_INPUT_TOPIC = cfg.PREDICTION_INPUT_TOPIC;
        this.PREDICTION_OUTPUT_TOPIC = cfg.PREDICTION_OUTPUT_TOPIC;

        System.out.println("Coordinator: " + PREDICTION_INPUT_TOPIC + ", " + PREDICTION_OUTPUT_TOPIC);

        this.RUN_ID = cfg.RUN_ID;

        this.FULLY_INFORMED = cfg.FULLY_INFORMED;
        this.DEBUG_KAFKA = cfg.DEBUG_KAFKA;

        this.logger = CustomLogger.getCoordinatorInstance();

        this.globalModel = Dl4jModelFactory.createModel();
        this.globalStats = new Stats();     

        this.predictor = BatchPrediction.getCoordinatorInstance(globalModel, globalStats);

        this.DATASET = cfg.DATASET;

        System.out.println("Running on Dataset: " + this.DATASET);

    }

    @Override
    public void run() {

        System.out.println("Coordinator started with RUN_ID: " + RUN_ID);

        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-coordinator-" + RUN_ID);
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1");
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "latest");
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false"); // not effective Kafka Streams commit by itself. It works only for plain
                                                                      // KafkaConsumers/KafkaProducers
        // props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        // props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        // props.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 0);
        // This controls how often Kafka Streams commits processing progress and flushes its internal caches.

        props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "3"); // 3 Threads since we have 3 Tasks {Task 0, Task 1, Task 2}

        // Serde<PBestUpdate> pBestSerde = new JsonSerde<>(PBestUpdate.class);

        CoordinatorControl control = CoordinatorControl.getInstance();

        StreamsBuilder builder = new StreamsBuilder();

        // Task 0 =======================================================================================================
        // input stream 5

        KStream<String, String> local_weights_stream = builder.stream(
            LOCAL_WEIGHTS_TOPIC,
            Consumed.with(Serdes.String(), Serdes.String())
        );

        local_weights_stream.process(() -> new CoordinatorProcessor(globalModel, globalStats));

        // Task 1 =======================================================================================================
        // input stream 3 and output stream 6

        if(1 == 2) { // for debuggging purposes

            KStream<String, String> pBest_weights_stream = builder.stream(
                PBEST_WEIGHTS_TOPIC,
                Consumed.with(Serdes.String(), Serdes.String())
            );

            pBest_weights_stream
                .process(() -> new CoordinatorProcessor(globalModel, globalStats))
                .to(GLOBAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), Serdes.String()));

        } else {
            
            if(FULLY_INFORMED != true) {

                KStream<String, String> pBestJsonStream = builder.stream(
                    PBEST_WEIGHTS_TOPIC,
                    Consumed.with(Serdes.String(), Serdes.String())
                )
                .peek((k, json) -> {
                    // logger.log("New gBest from worker JSON: " + json);
                });

                KTable<String, String> gBestTable = pBestJsonStream
                    .groupByKey()
                    .aggregate(
                        () -> null,     // initial aggregate = null (no gBest yet)
                        (key, newJson, aggJson) -> {
                            if (aggJson == null) return newJson;

                            try {
                                Map<String, Object> newMsg = MAPPER.readValue(newJson, new TypeReference<Map<String, Object>>() {});
                                Map<String, Object> oldMsg = MAPPER.readValue(aggJson, new TypeReference<Map<String, Object>>() {});

                                // double newAcc = ((Number) newMsg.get("accuracy")).doubleValue();
                                // double oldAcc = ((Number) oldMsg.get("accuracy")).doubleValue();

                                double newLoss = ((Number) newMsg.get("loss")).doubleValue();
                                double oldLoss = ((Number) oldMsg.get("loss")).doubleValue();

                                if(oldLoss > newLoss) {
                                    return newJson;
                                }

                                return aggJson; // means keep aggJson as the current aggregate

                            } catch (Exception e) {
                                e.printStackTrace();
                                return aggJson; 
                            }
                        },
                        Materialized.<String, String, KeyValueStore<Bytes, byte[]>>as("gBestStore")
                            .withKeySerde(Serdes.String())
                            .withValueSerde(Serdes.String())
                            .withCachingDisabled() 
                )
                .suppress(Suppressed.untilTimeLimit(
                    Duration.ofSeconds(1), // flush every one second
                    Suppressed.BufferConfig.unbounded()
                ));

                gBestTable
                    .toStream()
                    .mapValues(json -> {
                        try {
                            Map<String, Object> msg = MAPPER.readValue(json, new TypeReference<Map<String, Object>>() {});

                            payload.put("id_worker", msg.get("id_worker"));
                            payload.put("accuracy", msg.get("accuracy"));
                            payload.put("pBestMsgIndex", msg.get("pBestMsgIndex"));
                            payload.put("gBestWeights", msg.get("pBestWeights"));

                            return MAPPER.writeValueAsString(payload);

                        } catch (Exception e) {
                            e.printStackTrace();
                            return null;
                        }
                    })
                    .filter((k, v) -> v != null)
                    .peek((k, json) -> {
                        // logger.log("New gBest JSON: " + json);
                    })
                    .to(GLOBAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), Serdes.String())); 
            }
        }

        // Task 2 =======================================================================================================
        
        // KStream<String, String> prediction_stream = builder.stream(
        //     PREDICTION_INPUT_TOPIC,
        //     Consumed.with(Serdes.String(), Serdes.String())
        // );

        // prediction_stream
        //     .transformValues(() -> new PredictSingle(control))
        //     .to(PREDICTION_OUTPUT_TOPIC, Produced.with(Serdes.String(), Serdes.String()));

        KStream<String, String> prediction_stream = builder.stream(
            PREDICTION_INPUT_TOPIC,
            Consumed.with(Serdes.String(), Serdes.String())
        );

        prediction_stream
            .peek((k, v) -> { System.out.println("New Prediction Record: " + v); })
            .mapValues(json -> predictor.predictSingle(json)) 
            .filter((k, v) -> v != null) 
            .to(PREDICTION_OUTPUT_TOPIC, Produced.with(Serdes.String(), Serdes.String()));
            
        // =======================================================================================================

        Topology topology = builder.build();

        KafkaStreams streams = new KafkaStreams(topology, props);

        CountDownLatch latch = new CountDownLatch(1);

        streams.setUncaughtExceptionHandler((Thread t, Throwable e) -> {  // handling exceptions
            Throwable cause = e;

            if (e instanceof org.apache.kafka.streams.errors.StreamsException && e.getCause() != null) {
                cause = e.getCause();
            }

            if (cause instanceof CoordinatorProcessor.DesiredAccuracyReachedException) {
                System.out.println("[Coordinator] Stopping because desired accuracy was reached.");
            } else {
                System.out.println("[Coordinator] Uncaught exception in thread " + t.getName());
                e.printStackTrace();
            }
            try {
                streams.close();
            } finally {
                latch.countDown();
            }
        });

        // Watcher thread: waits for desired accuracy, then stops the streams ===========================================
        // Thread controlThread = new Thread(() -> {
        //     try {
        //         while (!control.isStopRequested()) {
        //             Thread.sleep(500); // poll every 500ms
        //         }
        //         System.out.println("[Coordinator] Stopping because desired accuracy was reached.");
        //         streams.close();
        //     } catch (InterruptedException ie) {
        //         Thread.currentThread().interrupt();
        //     } finally {
        //         latch.countDown();
        //     }
        // }, "coordinator-control-thread");

        // controlThread.setDaemon(true);
        // controlThread.start();

        // ===============================================================================================================

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {     // executes when doing Ctrl + C, SIGINT -> JVM -> addShutdownHook
            System.out.println("[Coordinator] Shutting down KafkaStreams");
            streams.close();
            latch.countDown();
        }));

        try {
            streams.start();

            System.out.println("[Coordinator] started.");
            if(DEBUG_KAFKA == true) {
                System.out.println("[Coordinator] Topology:\n" + topology.describe());

                try { 
                    Thread.sleep(2500); 
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

            latch.await(); 

        } catch (Throwable e) {
            System.out.println("[Coordinator] Error in KafkaStreams: " + e.getMessage());
            streams.close();
            latch.countDown();
        }

        System.out.println("[Coordinator] Exiting run()");

    }
}

// package pso;

// import org.apache.kafka.common.serialization.Serdes;
// import org.apache.kafka.common.serialization.Serde;
// import org.apache.kafka.streams.*;
// import org.apache.kafka.streams.kstream.Consumed;
// import org.apache.kafka.streams.kstream.Produced;
// import org.apache.kafka.clients.consumer.ConsumerConfig;
// import org.apache.kafka.streams.kstream.KStream;
// import org.apache.kafka.streams.kstream.KTable;
// import org.apache.kafka.streams.kstream.Materialized;
// import org.apache.kafka.streams.state.KeyValueStore;
// import org.apache.kafka.streams.state.StoreBuilder;
// import org.apache.kafka.streams.state.Stores;
// import org.apache.kafka.common.utils.Bytes;
// import org.apache.kafka.streams.kstream.Suppressed;
// import org.apache.kafka.streams.kstream.ValueTransformerWithKey;
// import org.apache.kafka.streams.processor.ThreadMetadata;
// import org.apache.kafka.streams.processor.TaskMetadata;
// import org.apache.kafka.streams.kstream.Grouped;
// import org.apache.kafka.streams.kstream.GlobalKTable;
// import org.apache.kafka.streams.Topology.AutoOffsetReset;

// import org.apache.kafka.clients.producer.ProducerConfig;

// import com.fasterxml.jackson.core.type.TypeReference;
// import com.fasterxml.jackson.databind.ObjectMapper;

// import java.util.Properties;
// import java.util.Map;
// import java.util.HashMap;
// import java.time.Duration;
// import java.util.concurrent.TimeUnit;
// import java.util.concurrent.atomic.AtomicInteger;
// import java.util.concurrent.CountDownLatch;

// import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

// import utils.*; 
// import state.*;
// import transformers.*;
// import message.data_message.*; 
// import message.weights_message.*; 

// import org.apache.kafka.streams.state.KeyValueStore;
// import org.apache.kafka.streams.state.Stores;
// import org.apache.kafka.streams.state.StoreBuilder;
// import java.util.concurrent.atomic.AtomicInteger;

// public class Coordinator implements Runnable {

//     private static Config cfg = Config.getInstance();
//     public final String DATASET = cfg.DATASET;
//     private final String DATA_TOPIC = cfg.DATA_TOPIC;
//     private final String TEST_TOPIC = cfg.TEST_TOPIC;
//     private final String PBEST_WEIGHTS_TOPIC = cfg.PBEST_WEIGHTS_TOPIC;
//     private final String LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
//     private final String GLOBAL_WEIGHTS_TOPIC = cfg.GLOBAL_WEIGHTS_TOPIC;
//     public final String PREDICTION_INPUT_TOPIC = cfg.PREDICTION_INPUT_TOPIC;
//     public final String PREDICTION_OUTPUT_TOPIC = cfg.PREDICTION_OUTPUT_TOPIC;

//     private final String RUN_ID = cfg.RUN_ID;
//     private final boolean FULLY_INFORMED = cfg.FULLY_INFORMED;
//     private final boolean DEBUG_KAFKA = cfg.DEBUG_KAFKA;
//     private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;

//     private final CustomLogger logger;

//     private final MultiLayerNetwork globalModel; // x_g , current model
//     private final MultiLayerNetwork bestGlobalModel;

//     private final BatchPrediction predictor;

//     private long t0 = System.nanoTime();
//     private long t1 = System.nanoTime();

//     final String TEST_STORE = "test-data-store";
//     private static final AtomicInteger INSTANCE_SEQ = new AtomicInteger(0);
//     private final int instanceNo = INSTANCE_SEQ.incrementAndGet();
//     private final String taskInstance = instanceNo + "@" + Integer.toHexString(System.identityHashCode(this));

//     private String taskTag = "task=UNKNOWN";

//     // ==============================================================================================================

//     public Coordinator() {

//         System.out.println("Coordinator: " + PREDICTION_INPUT_TOPIC + ", " + PREDICTION_OUTPUT_TOPIC);

//         this.logger = CustomLogger.getInstanceForCoordinator();

//         this.globalModel = Dl4jModelFactory.createModel();
//         this.bestGlobalModel = Dl4jModelFactory.createModel();

//         this.predictor = BatchPrediction.getInstanceForCoordinator(globalModel, bestGlobalModel, logger);

//         System.out.println("Running on Dataset: " + this.DATASET + ", with TEST_TOPIC: " + TEST_TOPIC);
//     }

//     // ==============================================================================================================

//     @Override
//     public void run() {

//         System.out.println("Coordinator started with RUN_ID: " + RUN_ID);

//         Properties props = new Properties();
//         props.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-coordinator-" + RUN_ID);
//         props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
//         props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "latest");
//         props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false"); 
//             // not effective Kafka Streams commit by itself. It works only for plain KafkaConsumers / KafkaProducers

//         // props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
//         // props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());
//         // props.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 0);
//         // This controls how often Kafka Streams commits processing progress and flushes its internal caches.

//         props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "3"); // 3 Threads since we have 3 Tasks {Task 0, Task 1, Task 2}
//         props.put(StreamsConfig.producerPrefix(ProducerConfig.MAX_REQUEST_SIZE_CONFIG), 5 * 1024 * 1024); // 5 MB

//         Serde<DataMessage> dataSerde = new DataMessageSerde();
//         Serde<WeightsMessage> weightsSerde = new WeightsMessageSerde();

//         CoordinatorControl control = CoordinatorControl.getInstance();

//         StreamsBuilder builder = new StreamsBuilder();

//         // =================================================================================================================
//         // GlobalKTable Task (Test Topic) ==================================================================================

//         GlobalKTable<String, DataMessage> testTable = builder.globalTable(
//             TEST_TOPIC,
//             Consumed.with(Serdes.String(), dataSerde)
//                 .withOffsetResetPolicy(Topology.AutoOffsetReset.EARLIEST),
//             Materialized.<String, DataMessage, KeyValueStore<Bytes, byte[]>>as(TEST_STORE)
//                 .withKeySerde(Serdes.String())
//                 .withValueSerde(dataSerde)
//         );

//         // Task 0 =======================================================================================================
//         // input stream 5

//         KStream<String, WeightsMessage> local_weights_stream = builder.stream(
//             LOCAL_WEIGHTS_TOPIC,
//             Consumed.with(Serdes.String(), weightsSerde)
//         );

//         local_weights_stream.process(() -> new CoordinatorProcessor(globalModel, bestGlobalModel, t0, t1, TEST_STORE));

//         // Task 1 =======================================================================================================
//         // input stream 3 and output stream 6 (ONLY IF FULLY_INFORMED == false)
        
//         if(FULLY_INFORMED == false) {    // if classical gBest PSO

//             if(1 == 2) {

//                 KStream<String, WeightsMessage> pBest_weights_stream = builder.stream(
//                     PBEST_WEIGHTS_TOPIC,
//                     Consumed.with(Serdes.String(), weightsSerde)
//                 );

//                 pBest_weights_stream
//                     .process(() -> new CoordinatorProcessor(globalModel, bestGlobalModel, t0, t1, TEST_STORE))      // doesnt actually edit the global model
//                     .to(GLOBAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));
            
//             // ===============================================================================================================

//             } else {

//                 StoreBuilder<KeyValueStore<String, Float>> gBestEmitStore =
//                     Stores.keyValueStoreBuilder(
//                         Stores.persistentKeyValueStore("gBestEmitStore"),
//                         Serdes.String(),
//                         Serdes.Float()
//                     );

//                 builder.addStateStore(gBestEmitStore);

//                 KStream<String, WeightsMessage> pBestStream = builder.stream(
//                     PBEST_WEIGHTS_TOPIC,
//                     Consumed.with(Serdes.String(), weightsSerde)
//                 ).peek((k, msg) -> {
//                     logger.log(taskInstance + " thread=" + Thread.currentThread().getName()
//             + "[pBest received] workerId: " + msg.workerId + ", msgIndex: " + msg.msgIndex+ ", acc: " + msg.accuracy
//                         + ", loss: " + msg.loss + ", weights: " + Dl4jParamUtils.sampleFlat(msg.weights, SAMPLING_CONSTANT)
//                     );
//                     // logger.log("[pBest received] workerId: " + msg.workerId + ", msgIndex: " + msg.msgIndex+ ", acc: " + msg.accuracy
//                     //     + ", loss: " + msg.loss + ", weights: " + Dl4jParamUtils.sampleFlat(msg.weights, SAMPLING_CONSTANT)
//                     // );                 
//                 });

//                 // pBestStream
//                 //     .process(() -> new GBestProcessor(logger), "gBestEmitStore")
//                 //     .to(GLOBAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));

//                 pBestStream
//                     .transform(() -> new GBestTransformer(logger), "gBestEmitStore")
//                     .to(GLOBAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));
//             }
//         }
        
//         // Task 2 ===============================================================================================================
//         // input stream 8 and output stream 9

//         KStream<String, DataMessage> prediction_stream = builder.stream(
//             PREDICTION_INPUT_TOPIC,
//             Consumed.with(Serdes.String(), dataSerde)
//         );

//         prediction_stream
//             .peek((k, v) -> { System.out.println("New Prediction Record: " + v); })
//             .mapValues(dataMessage -> predictor.predictSingleBest(dataMessage))   // simple transformer
//             .filter((k, v) -> v != null) 
//             .to(PREDICTION_OUTPUT_TOPIC, Produced.with(Serdes.String(), Serdes.String()));

//         // =================================================================================================================================

//         Topology topology = builder.build();

//         KafkaStreams streams = new KafkaStreams(topology, props);

//         CountDownLatch latch = new CountDownLatch(1);

//         streams.setUncaughtExceptionHandler((Thread t, Throwable e) -> {  // handling exceptions
//             Throwable cause = e;

//             if (e instanceof org.apache.kafka.streams.errors.StreamsException && e.getCause() != null) {
//                 cause = e.getCause();
//             }

//             if (cause instanceof CoordinatorProcessor.DesiredAccuracyReachedException) {
//                 System.out.println("[Coordinator] Stopping because desired accuracy was reached.");
//             } else {
//                 System.out.println("[Coordinator] Uncaught exception in thread " + t.getName());
//                 e.printStackTrace();
//             }
//             try {
//                 streams.close();
//             } finally {
//                 latch.countDown();
//             }
//         });

//         // ===============================================================================================================

//         Thread controlThread = new Thread(() -> {
//             try {
//                 while (!control.isStopRequested(-1)) {
//                     Thread.sleep(100);
//                 }
//                 System.out.println("[Coordinator] Stop requested, closing streams");
//                 streams.close();

//                 latch.countDown();

//                 t1 = System.nanoTime();
//                 final double seconds = (t1 - t0) / 1_000_000_000.0;

//                 System.out.printf("[Coordinator] Wall time: %.3f seconds%n", seconds);

//             } catch (InterruptedException ie) {
//                 Thread.currentThread().interrupt();
//             }

//         }, "coordinator-control-thread");

//         controlThread.setDaemon(true);
//         controlThread.start();

//         // ===============================================================================================================
//         // This runs on exception

//         Runtime.getRuntime().addShutdownHook(new Thread(() -> {     // executes when doing Ctrl + C, SIGINT -> JVM -> addShutdownHook
//             System.out.println("[Coordinator] Shutting down KafkaStreams");
//             streams.close();
//             latch.countDown();
//         }));

//         // ===============================================================================================================

//         try {
//             streams.start();

//             System.out.println("[Coordinator] started.");
//             if(DEBUG_KAFKA == true) {
//                 System.out.println("[Coordinator] Topology:\n" + topology.describe());

//                 try { 
//                     Thread.sleep(2500); 
//                 } catch (InterruptedException ignored) {
//                     System.out.println("Sleep failed");
//                 }

//                 for (ThreadMetadata tm : streams.localThreadsMetadata()) {
//                     System.out.println("Thread: " + tm.threadName() + " state=" + tm.threadState());

//                     for (TaskMetadata task : tm.activeTasks()) {
//                         System.out.println("  ACTIVE Task: " + task.taskId()
//                             + " partitions=" + task.topicPartitions());
//                     }
//                 }
//             }

//             latch.await(); // wait until Kafka Streams has consumed the test topic and you have exited 

//         } catch (Throwable e) {
//             System.out.println("[Coordinator] Error in KafkaStreams: " + e.getMessage());
//             streams.close();
//             latch.countDown();
//         }

//         System.out.println("[Coordinator] Exiting run()");

//     }
// }
        
//     // KStream<String, WeightsMessage> pBestJsonStream = builder.stream(
//     //     PBEST_WEIGHTS_TOPIC,
//     //     Consumed.with(Serdes.String(), weightsSerde)
//     // )
//     // .peek((k, msg) -> {
//     //     logger.log(
//     //         "[pBest received] workerId: " + msg.workerId + ", msgIndex: " + msg.msgIndex + ", acc: " + msg.accuracy +
//     //         ", loss: " + msg.loss +", weights: " + Dl4jParamUtils.sampleFlat(msg.weights, SAMPLING_CONSTANT)
//     //     );
//     // });

//     // KTable<String, WeightsMessage> gBestTable = pBestJsonStream
//     //     .groupByKey(Grouped.with(Serdes.String(), weightsSerde))
//     //     .aggregate(
//     //         () -> null,     // initial aggregate = null (no gBest yet)
//     //         (key, newMsg, aggMsg) -> {
//     //             if (aggMsg == null) {
//     //                 return newMsg;
//     //             }
                
//     //             float newLoss = newMsg.loss;
//     //             float oldLoss = aggMsg.loss;

//     //             if (newLoss < oldLoss) {
//     //                 return newMsg;
//     //             }
//     //             return aggMsg; // means keep aggMsg as the current aggregate

//     //         },
//     //         Materialized.<String, WeightsMessage, KeyValueStore<Bytes, byte[]>>as("gBestStore") 
//     //             .withKeySerde(Serdes.String())  // 0 interatction with CoordinatorProcessor, this only relays / updates the gBest for the workers
//     //             .withValueSerde(weightsSerde)   // the dtatestore is used here only, just to remember what the current aggregate is
//     //             .withCachingDisabled() 
//     // );
//     // // .suppress(Suppressed.untilTimeLimit(    // just buffers updates and only forwards the latest per key after 1 second.
//     // //     Duration.ofSeconds(1),              // flush every one second
//     // //     Suppressed.BufferConfig.unbounded()
//     // // ));

//     // gBestTable
//     //     .toStream()
//     //     .filter((k, v) -> v != null)
//     //     .peek((k, msg) -> {
//     //         logger.log(
//     //             "[gBest sended] workerId: " + msg.workerId + ", msgIndex: " + msg.msgIndex + 
//     //             ", acc: " + msg.accuracy + ", loss: " + msg.loss + ", weights: " + 
//     //             Dl4jParamUtils.sampleFlat(msg.weights, SAMPLING_CONSTANT)
//     //         );
//     //     })
//     //     .to(GLOBAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde)); 



package pso;

import org.apache.kafka.common.serialization.Serde;
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.producer.ProducerConfig;

import org.apache.kafka.common.utils.Bytes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.Topology;

import org.apache.kafka.streams.kstream.Consumed;
import org.apache.kafka.streams.kstream.GlobalKTable;
import org.apache.kafka.streams.kstream.KStream;
import org.apache.kafka.streams.kstream.Materialized;
import org.apache.kafka.streams.kstream.Produced;

import org.apache.kafka.streams.processor.ThreadMetadata;
import org.apache.kafka.streams.processor.TaskMetadata;

import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.streams.state.StoreBuilder;
import org.apache.kafka.streams.state.Stores;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.time.Duration;
import java.util.Properties;
import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

import utils.*; 
import state.*;
import transformers.*;
import message.data_message.*; 
import message.weights_message.*; 
import transformers.*;

public class Coordinator implements Runnable {

    private static final Config cfg = Config.getInstance();

    public final String DATASET = cfg.DATASET;
    private final String DATA_TOPIC = cfg.DATA_TOPIC;
    private final String TEST_TOPIC = cfg.TEST_TOPIC;
    private final String PBEST_WEIGHTS_TOPIC = cfg.PBEST_WEIGHTS_TOPIC;
    private final String LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
    private final String GLOBAL_WEIGHTS_TOPIC = cfg.GLOBAL_WEIGHTS_TOPIC;
    public final String PREDICTION_INPUT_TOPIC = cfg.PREDICTION_INPUT_TOPIC;
    public final String PREDICTION_OUTPUT_TOPIC = cfg.PREDICTION_OUTPUT_TOPIC;

    private final String RUN_ID = cfg.RUN_ID;
    private final boolean FULLY_INFORMED = cfg.FULLY_INFORMED;
    private final boolean DEBUG_KAFKA = cfg.DEBUG_KAFKA;

    private final CustomLogger logger;

    private final MultiLayerNetwork globalModel;
    private final MultiLayerNetwork bestGlobalModel;
    private final BatchPrediction predictor;

    private long t0 = System.nanoTime();
    private long t1 = System.nanoTime();
    private double lastActivitySeconds = 0.0;

    private static final String TEST_STORE = "test-data-store";

    private static final AtomicInteger INSTANCE_SEQ = new AtomicInteger(0);
    private final int instanceNo = INSTANCE_SEQ.incrementAndGet();
    private final String instanceTag = "Coordinator@" + instanceNo + "#" + Integer.toHexString(System.identityHashCode(this));

    public Coordinator() {
        this.logger = CustomLogger.getInstanceForCoordinator();

        this.globalModel = Dl4jModelFactory.createModel();
        this.bestGlobalModel = Dl4jModelFactory.createModel();
        this.predictor = BatchPrediction.getInstanceForCoordinator(globalModel, bestGlobalModel, logger);

        System.out.println("Running on Dataset: " + DATASET + ", TEST_TOPIC: " + TEST_TOPIC);
        System.out.println("Coordinator topics: PRED_IN=" + PREDICTION_INPUT_TOPIC + ", PRED_OUT=" + PREDICTION_OUTPUT_TOPIC);
    }

    @Override
    public void run() {

        this.t0 = System.nanoTime();
        this.t1 = System.nanoTime();

        System.out.println(instanceTag + " started with RUN_ID: " + RUN_ID);

        // --- Serdes ---
        Serde<DataMessage> dataSerde = new DataMessageSerde();
        Serde<WeightsMessage> weightsSerde = new WeightsMessageSerde();

        // --- Shared base properties (we will clone and override app.id / threads per instance) ---
        Properties baseProps = new Properties();
        baseProps.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        baseProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "latest");
        baseProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");
        baseProps.put(StreamsConfig.producerPrefix(ProducerConfig.MAX_REQUEST_SIZE_CONFIG), 5 * 1024 * 1024); // 5MB

        // MAIN instance props
        Properties mainProps = new Properties();
        mainProps.putAll(baseProps);
        mainProps.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-coordinator-" + RUN_ID);
        mainProps.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "2");

        // GBEST instance props (separate app.id!)
        Properties gbestProps = new Properties();
        gbestProps.putAll(baseProps);
        gbestProps.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-gbest-relay-" + RUN_ID);
        gbestProps.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1");

        // Build topologies =============================================================
        Topology mainTopology = buildMainTopology(dataSerde, weightsSerde);
        Topology gbestTopology = null;
        if (!FULLY_INFORMED) {
            gbestTopology = buildGBestRelayTopology(weightsSerde);
        }

        // Create streams Instances =============================================================
        KafkaStreams mainStreams = new KafkaStreams(mainTopology, mainProps);
        KafkaStreams gbestStreams = (!FULLY_INFORMED) ? new KafkaStreams(gbestTopology, gbestProps) : null;

        CoordinatorControl control = CoordinatorControl.getInstance();
        CountDownLatch latch = new CountDownLatch(1);

        // --- Exception handler: close both ---
        mainStreams.setUncaughtExceptionHandler((Thread t, Throwable e) -> {
            Throwable cause = e;

            if (e instanceof org.apache.kafka.streams.errors.StreamsException && e.getCause() != null) {
                cause = e.getCause();
            }

            System.out.println("[Coordinator] Uncaught exception in thread " + t.getName());
            cause.printStackTrace();

            try {
                safeClose(mainStreams);
                safeClose(gbestStreams);
            } finally {
                latch.countDown();
            }
        });

        if (gbestStreams != null) {
            gbestStreams.setUncaughtExceptionHandler((Thread t, Throwable e) -> {
                System.out.println("[GBestRelay] Uncaught exception in thread " + t.getName());
                e.printStackTrace();

                try {
                    safeClose(gbestStreams);
                    // You can decide whether to stop MAIN too; I usually do:
                    safeClose(mainStreams);
                } finally {
                    latch.countDown();
                }
            });
        }

        // Control thread =============================================================

        Thread controlThread = new Thread(() -> {
            try {
                while (!control.isStopRequested(-1)) {
                    Thread.sleep(100);
                }
                System.out.println("[Coordinator] Stop requested, closing streams");

                safeClose(mainStreams);
                safeClose(gbestStreams);

                latch.countDown();

                t1 = System.nanoTime();
                final double seconds = (t1 - t0) / 1_000_000_000.0;
                System.out.printf("[Coordinator] Wall time: %.3f seconds%n", seconds);

            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
        }, "coordinator-control-thread");
        controlThread.setDaemon(true);
        controlThread.start();

        // --- Shutdown hook ---
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[Coordinator] Shutting down KafkaStreams");
            safeClose(mainStreams);
            safeClose(gbestStreams);
            latch.countDown();
        }));

        // --- Start ---
        try {
            mainStreams.start();
            if (gbestStreams != null) gbestStreams.start();

            System.out.println("[Coordinator] MAIN started. FULLY_INFORMED=" + FULLY_INFORMED);
            if (gbestStreams != null) {
                System.out.println("[Coordinator] GBEST-RELAY started.");
            }

            if (DEBUG_KAFKA) {
                System.out.println("\n[MAIN Topology]\n" + mainTopology.describe());
                if (gbestTopology != null) {
                    System.out.println("\n[GBEST Topology]\n" + gbestTopology.describe());
                }

                // wait a bit for tasks to initialize
                try { Thread.sleep(2500); } catch (InterruptedException ignored) {}

                System.out.println("\n[MAIN Threads/Tasks]");
                printThreadsAndTasks(mainStreams);

                if (gbestStreams != null) {
                    System.out.println("\n[GBEST Threads/Tasks]");
                    printThreadsAndTasks(gbestStreams);
                }
            }

            latch.await(); // block until shutdown

        } catch (Throwable e) {
            System.out.println("[Coordinator] Error in KafkaStreams: " + e.getMessage());
            e.printStackTrace();
            safeClose(mainStreams);
            safeClose(gbestStreams);
            latch.countDown();
        }

        System.out.println("[Coordinator] Exiting run()");
    }

    // ==========================================================================================================
    // MAIN TOPOLOGY
    // ==========================================================================================================

    private Topology buildMainTopology(Serde<DataMessage> dataSerde, Serde<WeightsMessage> weightsSerde) {

        StreamsBuilder builder = new StreamsBuilder();

        // Loads the TEST_TOPIC ===========================================================================================

        GlobalKTable<String, DataMessage> testTable = builder.globalTable(
            TEST_TOPIC,
            Consumed.with(Serdes.String(), dataSerde)
                .withOffsetResetPolicy(Topology.AutoOffsetReset.EARLIEST),
            Materialized.<String, DataMessage, KeyValueStore<Bytes, byte[]>>as(TEST_STORE)
                .withKeySerde(Serdes.String())
                .withValueSerde(dataSerde)
        );

        // Federrated Learning Monitoring Pipeline =========================================================================

        KStream<String, WeightsMessage> localWeightsStream = builder.stream(
            LOCAL_WEIGHTS_TOPIC,
            Consumed.with(Serdes.String(), weightsSerde)
        );

        localWeightsStream.process(() -> new CoordinatorProcessor(globalModel, bestGlobalModel, t0, t1, TEST_STORE));

        // Inference Task ==================================================================================================
        
        KStream<String, DataMessage> predictionStream = builder.stream(
            PREDICTION_INPUT_TOPIC,
            Consumed.with(Serdes.String(), dataSerde)
        );

        predictionStream
            .mapValues(dm -> predictor.predictSingleBest(dm))
            .filter((k, v) -> v != null)
            .to(PREDICTION_OUTPUT_TOPIC, Produced.with(Serdes.String(), Serdes.String()));

        return builder.build();
    }

    // ==========================================================================================================
    // GBEST RELAY TOPOLOGY (separate instance)
    // ==========================================================================================================

    private Topology buildGBestRelayTopology(Serde<WeightsMessage> weightsSerde) {

        StreamsBuilder builder = new StreamsBuilder();

        // State store ONLY for this instance
        StoreBuilder<KeyValueStore<String, Float>> gBestEmitStore =
            Stores.keyValueStoreBuilder(
                Stores.persistentKeyValueStore("gBestEmitStore"),
                Serdes.String(),
                Serdes.Float()
            );

        builder.addStateStore(gBestEmitStore);

        KStream<String, WeightsMessage> pBestStream = builder.stream(
            PBEST_WEIGHTS_TOPIC,
            Consumed.with(Serdes.String(), weightsSerde)
        );

        pBestStream.peek((k, msg) -> {
            updateTime();
            logger.log(lastActivitySeconds +", [pBest received] workerId = " + msg.workerId + ", msgindex: " + msg.msgIndex
                + ", accuracy: " + msg.accuracy + ", loss: " + msg.loss);
        });

        pBestStream
            .transform(() -> new GBestTransformer(logger, t0), "gBestEmitStore")
            .to(GLOBAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));

        return builder.build();
    }

    // ==========================================================================================================
    // Helpers
    // ==========================================================================================================

    private static void safeClose(KafkaStreams streams) {
        if (streams == null) return;
        try {
            streams.close(Duration.ofSeconds(5));
        } catch (Exception ignored) {}
    }

    private static void printThreadsAndTasks(KafkaStreams streams) {
        for (ThreadMetadata tm : streams.localThreadsMetadata()) {
            System.out.println("Thread: " + tm.threadName() + " state=" + tm.threadState());
            for (TaskMetadata task : tm.activeTasks()) {
                System.out.println("  ACTIVE Task: " + task.taskId() + " partitions=" + task.topicPartitions());
            }
        }
    }

    private void updateTime() {
        lastActivitySeconds = Math.round(((System.nanoTime() - t0) / 1_000_000_000.0) * 1000.0) / 1000.0;
    }
}

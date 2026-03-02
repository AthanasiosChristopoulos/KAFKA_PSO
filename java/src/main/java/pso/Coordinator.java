
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
import org.nd4j.common.primitives.Pair;

import dl4j_models.Dl4jModelFactory;
import dl4j_models.PsoModel;

import java.time.Duration;
import java.util.Properties;
import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

import utils.*; 
import state.*;
import experimentation.*;
import transformers.*;
import message.data_message.*; 
import message.weights_message.*; 

public class Coordinator implements Runnable {

    private static final Config cfg = Config.getInstance();

    public final String DATASET = cfg.DATASET;
    private final String DATA_TOPIC = cfg.DATA_TOPIC;
    private final String TEST_TOPIC = cfg.TEST_TOPIC;
    private final String PBEST_WEIGHTS_TOPIC = cfg.PBEST_WEIGHTS_TOPIC;
    private final String LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
    private final String GPEST_WEIGHTS_TOPIC = cfg.GPEST_WEIGHTS_TOPIC;
    public final String PREDICTION_INPUT_TOPIC = cfg.PREDICTION_INPUT_TOPIC;
    public final String PREDICTION_OUTPUT_TOPIC = cfg.PREDICTION_OUTPUT_TOPIC;

    private final String RUN_ID = cfg.RUN_ID;
    private final boolean FULLY_INFORMED = cfg.FULLY_INFORMED;
    private final boolean DEBUG_KAFKA = cfg.DEBUG_KAFKA;
    private final boolean ENABLE_NEIGHBORHOODS = cfg.ENABLE_NEIGHBORHOODS;

    private final CustomLogger logger;

    private final PsoModel globalModel;
    private final PsoModel bestGlobalModel;
    private PsoModel preTrainedModel;
    private final BatchPrediction predictor;

    private long t0 = System.nanoTime();
    private long t1 = System.nanoTime();
    private double lastActivitySeconds = 0.0;

    private static final String TEST_STORE = "test-data-store";

    private static final AtomicInteger INSTANCE_SEQ = new AtomicInteger(0);
    private final int instanceNo = INSTANCE_SEQ.incrementAndGet();
    private final String instanceTag = "Coordinator@" + instanceNo + "#" + Integer.toHexString(System.identityHashCode(this));

    private Pair<PsoModel, Integer> pair_best;

    private int start;

    private final MetricsCollector collector; 

    // ====================================================================================================================================

    public Coordinator(MetricsCollector collector) {

        this.logger = CustomLogger.getInstanceForCoordinator();

        // Need to do the instancing here the transformers constructor runs many times from different tasks
        this.globalModel = Dl4jModelFactory.createModel(-1, false).getFirst();
        System.out.println("Model Summary ===========================================");
        if(logger.isEnabled(2)) logger.log("Model Summary ===========================================");
        if(logger.isEnabled(2)) logger.log(this.globalModel.summary());

        pair_best = Dl4jModelFactory.createModel(-1, false);
        this.bestGlobalModel = pair_best.getFirst();
        this.start = pair_best.getSecond();

        if(cfg.USING_PRETRAINED_MODEL || cfg.FREEZE) {
            this.preTrainedModel = Dl4jModelFactory.createModel(-1, true).getFirst();
            if(logger.isEnabled(2)) logger.log("Pretrained Summary ===========================================");
            if(logger.isEnabled(2)) logger.log(this.preTrainedModel.summary());
        }

        System.out.println(this.globalModel.summary());

        this.predictor = BatchPrediction.getInstanceForCoordinator(globalModel, bestGlobalModel, logger);

        System.out.println("Running on Dataset: " + DATASET + ", TEST_TOPIC: " + TEST_TOPIC);
        System.out.println("Coordinator topics: PRED_IN=" + PREDICTION_INPUT_TOPIC + ", PRED_OUT=" + PREDICTION_OUTPUT_TOPIC);

        this.collector = collector;
    }

    // ====================================================================================================================================

    @Override
    public void run() {

        this.t0 = System.nanoTime();
        this.t1 = System.nanoTime();

        System.out.println(instanceTag + " started with RUN_ID: " + RUN_ID);

        // Serdes
        Serde<DataMessage> dataSerde = new DataMessageSerde();
        Serde<WeightsMessage> weightsSerde = new WeightsMessageSerde();

        // Shared base properties (we will clone and override app.id / threads per instance)
        Properties baseProps = new Properties();
        baseProps.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, cfg.KAFKA_HOST);
        baseProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "latest");
        baseProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");
        baseProps.put(StreamsConfig.producerPrefix(ProducerConfig.MAX_REQUEST_SIZE_CONFIG), 5 * 1024 * 1024); // 5MB
        baseProps.put(StreamsConfig.producerPrefix(ProducerConfig.LINGER_MS_CONFIG), 0);

        // MAIN instance props fdfdf
        Properties mainProps = new Properties();
        mainProps.putAll(baseProps);
        mainProps.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-coordinator-" + RUN_ID);
        mainProps.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "2");
        mainProps.put(StreamsConfig.STATE_DIR_CONFIG, cfg.KAFKA_TMP_DIR + "/main-" + RUN_ID);

        // GBEST instance props (separate app.id!)
        Properties gbestProps = new Properties();
        gbestProps.putAll(baseProps);
        gbestProps.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-gbest-relay-" + RUN_ID);
        gbestProps.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1");
        gbestProps.put(StreamsConfig.STATE_DIR_CONFIG, cfg.KAFKA_TMP_DIR + "/gbest-" + RUN_ID);

        // Build topologies =============================================================

        Topology mainTopology = buildMainTopology(dataSerde, weightsSerde);
        Topology gbestTopology = null;
        if (!ENABLE_NEIGHBORHOODS && !FULLY_INFORMED) {
            gbestTopology = buildGBestRelayTopology(weightsSerde);
        }

        // Create streams Instances =============================================================

        KafkaStreams mainStreams = new KafkaStreams(mainTopology, mainProps);
        KafkaStreams gbestStreams = (gbestTopology != null) ? new KafkaStreams(gbestTopology, gbestProps) : null;

        CoordinatorControl control = CoordinatorControl.getInstance();
        CountDownLatch latch = new CountDownLatch(1);

        // Exception handler: close both

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
                    Thread.sleep(50); 
                }

                System.out.println("[Coordinator] Stop requested, closing streams");

                safeClose(mainStreams);
                safeClose(gbestStreams);

                latch.countDown();

                t1 = System.nanoTime();
                final double seconds = (t1 - t0) / 1_000_000_000.0;

                if(collector != null) {
                    collector.reportCoordinatorDone(new CoordinatorMetrics(seconds, 
                        control.getBestGlobalModelAccuracy(), control.getBestGlobalModelLoss()));
                }

                System.out.println("[Coordinator] Final (Best) Results: Training Accuracy: " + control.getBestTrainingAccuracy()
                     + ", Test Accuracy:" + control.getBestGlobalModelAccuracy());

                System.out.printf("[Coordinator] Elapsed time: %.3f seconds%n", seconds);

            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            }
        }, "coordinator-control-thread");
        controlThread.setDaemon(true);
        controlThread.start();

        // Shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[Coordinator] Shutting down KafkaStreams");
            safeClose(mainStreams);
            safeClose(gbestStreams);
            latch.countDown();
        }));

        // Start

        try {
            mainStreams.start();
            if (gbestStreams != null) gbestStreams.start();

            System.out.println("[Coordinator] started. FULLY_INFORMED = " + FULLY_INFORMED);

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
        
        localWeightsStream.process(() -> new CoordinatorProcessor(globalModel, bestGlobalModel, t0, t1, 
                TEST_STORE, this.preTrainedModel, this.start));

        // Inference Task ==================================================================================================
        // KStream<String, DataMessage> predictionStream = builder.stream(
        //     PREDICTION_INPUT_TOPIC,
        //     Consumed.with(Serdes.String(), dataSerde)
        // );

        // predictionStream
        //     .mapValues(dm -> predictor.predictSingleBest(dm))
        //     .filter((k, v) -> v != null)
        //     .to(PREDICTION_OUTPUT_TOPIC, Produced.with(Serdes.String(), Serdes.String()));

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
                Stores.inMemoryKeyValueStore("gBestEmitStore"),     // Stores.persistentKeyValueStore("gBestEmitStore"), KeyValueStore is an interface
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
            if (logger.isEnabled(1)) logger.log(lastActivitySeconds +", [pBest received] workerId = " + msg.workerId + ", msgindex: " + msg.msgIndex
                + ", accuracy: " + msg.accuracy + ", loss: " + msg.loss);
        });

        pBestStream
            .transform(() -> new GBestTransformer(logger, t0), "gBestEmitStore")
            .to(GPEST_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));

        return builder.build();
    }

    // ==========================================================================================================
    // Helpers
    // ==========================================================================================================

    private static void safeClose(KafkaStreams streams) {
        if (streams == null) return;
        try {
            streams.close(Duration.ofSeconds(5));
        } catch (Exception ex) { 
            ex.printStackTrace(); 
        }

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

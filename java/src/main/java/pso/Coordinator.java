package pso;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.common.serialization.Serde;
import org.apache.kafka.streams.*;
import org.apache.kafka.streams.kstream.Consumed;
import org.apache.kafka.streams.kstream.Produced;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.streams.kstream.KStream;
import org.apache.kafka.streams.kstream.KTable;
import org.apache.kafka.streams.kstream.Materialized;
import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.common.utils.Bytes;
import org.apache.kafka.streams.kstream.Suppressed;
import org.apache.kafka.streams.processor.ThreadMetadata;
import org.apache.kafka.streams.processor.TaskMetadata;
import org.apache.kafka.streams.kstream.Grouped;
import org.apache.kafka.streams.kstream.GlobalKTable;
import org.apache.kafka.streams.Topology.AutoOffsetReset;

import org.apache.kafka.clients.producer.ProducerConfig;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.Properties;
import java.util.Map;
import java.util.HashMap;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

import java.util.concurrent.CountDownLatch;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import utils.*; 
import state.*; 
import message.data_message.*; 
import message.weights_message.*; 


public class Coordinator implements Runnable {

    private static Config cfg = Config.getInstance();
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
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;

    private final CustomLogger logger;

    private final MultiLayerNetwork globalModel; // x_g , current model
    private final MultiLayerNetwork bestGlobalModel;

    private final BatchPrediction predictor;

    private long t0 = System.nanoTime();
    private long t1 = System.nanoTime();

    final String TEST_STORE = "test-data-store";

    // ==============================================================================================================

    public Coordinator() {

        System.out.println("Coordinator: " + PREDICTION_INPUT_TOPIC + ", " + PREDICTION_OUTPUT_TOPIC);

        this.logger = CustomLogger.getInstanceForCoordinator();

        this.globalModel = Dl4jModelFactory.createModel();
        this.bestGlobalModel = Dl4jModelFactory.createModel();

        this.predictor = BatchPrediction.getInstanceForCoordinator(globalModel, bestGlobalModel, logger);

        System.out.println("Running on Dataset: " + this.DATASET);
    }

    // ==============================================================================================================

    @Override
    public void run() {

        System.out.println("Coordinator started with RUN_ID: " + RUN_ID);

        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-coordinator-" + RUN_ID);
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "latest");
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false"); 
            // not effective Kafka Streams commit by itself. It works only for plain KafkaConsumers / KafkaProducers

        // props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        // props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        // props.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 0);
        // This controls how often Kafka Streams commits processing progress and flushes its internal caches.

        props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "3"); // 3 Threads since we have 3 Tasks {Task 0, Task 1, Task 2}
        props.put(StreamsConfig.producerPrefix(ProducerConfig.MAX_REQUEST_SIZE_CONFIG), 5 * 1024 * 1024); // 5 MB

        Serde<DataMessage> dataSerde = new DataMessageSerde();
        Serde<WeightsMessage> weightsSerde = new WeightsMessageSerde();

        CoordinatorControl control = CoordinatorControl.getInstance();

        StreamsBuilder builder = new StreamsBuilder();

        // GlobalKTable Task (Test Topic) ==================================================================================

        GlobalKTable<String, DataMessage> testTable = builder.globalTable(
            TEST_TOPIC,
            Consumed.with(Serdes.String(), dataSerde)
                .withOffsetResetPolicy(Topology.AutoOffsetReset.EARLIEST),
            Materialized.<String, DataMessage, KeyValueStore<Bytes, byte[]>>as(TEST_STORE)
                .withKeySerde(Serdes.String())
                .withValueSerde(dataSerde)
        );

        // Task 0 =======================================================================================================
        // input stream 5

        KStream<String, WeightsMessage> local_weights_stream = builder.stream(
            LOCAL_WEIGHTS_TOPIC,
            Consumed.with(Serdes.String(), weightsSerde)
        );

        local_weights_stream.process(() -> new CoordinatorProcessor(globalModel, bestGlobalModel, t0, t1, TEST_STORE));

        // Task 1 =======================================================================================================
        // input stream 3 and output stream 6 (ONLY IF FULLY_INFORMED == false)
        
        if(FULLY_INFORMED != true) {    // if classical gBest PSO

            if(1 == 1) { // for debuggging purposes

                KStream<String, WeightsMessage> pBest_weights_stream = builder.stream(
                    PBEST_WEIGHTS_TOPIC,
                    Consumed.with(Serdes.String(), weightsSerde)
                );

                pBest_weights_stream
                    .process(() -> new CoordinatorProcessor(globalModel, bestGlobalModel, t0, t1, TEST_STORE))      // doesnt actually edit the global model
                    .to(GLOBAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));

            } else {

                KStream<String, WeightsMessage> pBestJsonStream = builder.stream(
                    PBEST_WEIGHTS_TOPIC,
                    Consumed.with(Serdes.String(), weightsSerde)
                )
                .peek((k, msg) -> {
                    logger.log(
                        "[pBest received] workerId: " + msg.idWorker + ", msgIndex: " + msg.msgIndex + ", acc: " + msg.accuracy +
                        ", loss: " + msg.loss +", weights: " + Dl4jParamUtils.sampleFlat(msg.weights, SAMPLING_CONSTANT)
                    );
                });

                KTable<String, WeightsMessage> gBestTable = pBestJsonStream
                    .groupByKey(Grouped.with(Serdes.String(), weightsSerde))
                    .aggregate(
                        () -> null,     // initial aggregate = null (no gBest yet)
                        (key, newMsg, aggMsg) -> {
                            if (aggMsg == null) {
                                return newMsg;
                            }
                            
                            float newLoss = newMsg.loss;
                            float oldLoss = aggMsg.loss;

                            if (newLoss < oldLoss) {
                                return newMsg;
                            }
                            return aggMsg; // means keep aggMsg as the current aggregate

                        },
                        Materialized.<String, WeightsMessage, KeyValueStore<Bytes, byte[]>>as("gBestStore")
                            .withKeySerde(Serdes.String())
                            .withValueSerde(weightsSerde)
                            .withCachingDisabled() 
                )
                .suppress(Suppressed.untilTimeLimit(    // just buffers updates and only forwards the latest per key after 1 second.
                    Duration.ofSeconds(1),              // flush every one second
                    Suppressed.BufferConfig.unbounded()
                ));

                gBestTable
                    .toStream()
                    .filter((k, v) -> v != null)
                    .peek((k, msg) -> {
                        logger.log(
                            "[gBest sended] workerId: " + msg.idWorker + ", msgIndex: " + msg.msgIndex + 
                            ", acc: " + msg.accuracy + ", loss: " + msg.loss + ", weights: " + 
                            Dl4jParamUtils.sampleFlat(msg.weights, SAMPLING_CONSTANT)
                        );
                    })
                    .to(GLOBAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde)); 
            }
        }
        
        // Task 2 ===============================================================================================================
        // input stream 8 and output stream 9

        KStream<String, DataMessage> prediction_stream = builder.stream(
            PREDICTION_INPUT_TOPIC,
            Consumed.with(Serdes.String(), dataSerde)
        );

        prediction_stream
            .peek((k, v) -> { System.out.println("New Prediction Record: " + v); })
            .mapValues(dataMessage -> predictor.predictSingleBest(dataMessage))   // simple transformer
            .filter((k, v) -> v != null) 
            .to(PREDICTION_OUTPUT_TOPIC, Produced.with(Serdes.String(), Serdes.String()));

        // =================================================================================================================================

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

        // ===============================================================================================================

        Thread controlThread = new Thread(() -> {
            try {
                while (!control.isStopRequested(-1)) {
                    Thread.sleep(100);
                }
                System.out.println("[Coordinator] Stop requested, closing streams");
                streams.close();

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

        // ===============================================================================================================
        // This runs on exception

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {     // executes when doing Ctrl + C, SIGINT -> JVM -> addShutdownHook
            System.out.println("[Coordinator] Shutting down KafkaStreams");
            streams.close();
            latch.countDown();
        }));

        // ===============================================================================================================

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

            latch.await(); // wait until Kafka Streams has consumed the test topic and you have exited 

        } catch (Throwable e) {
            System.out.println("[Coordinator] Error in KafkaStreams: " + e.getMessage());
            streams.close();
            latch.countDown();
        }

        System.out.println("[Coordinator] Exiting run()");

    }
}

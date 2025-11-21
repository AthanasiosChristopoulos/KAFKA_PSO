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

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.Properties;
import java.util.Map;
import java.util.HashMap;

import java.util.concurrent.CountDownLatch;

import utils.*; 

public class Coordinator implements Runnable {

    private final String PBEST_WEIGHTS_TOPIC;
    private final String LOCAL_WEIGHTS_TOPIC;
    private final String GLOBAL_WEIGHTS_TOPIC;
    private final String RUN_ID;
    private final String FULLY_INFORMED;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final CustomLogger logger;

    public Coordinator() {

        Config cfg = Config.get();
        this.RUN_ID = cfg.RUN_ID;
        this.PBEST_WEIGHTS_TOPIC = cfg.PBEST_WEIGHTS_TOPIC;
        this.LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
        this.GLOBAL_WEIGHTS_TOPIC = cfg.GLOBAL_WEIGHTS_TOPIC;
        this.FULLY_INFORMED = cfg.FULLY_INFORMED;

        this.logger = CustomLogger.getCoordinatorInstance();
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
        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

        // Serde<PBestUpdate> pBestSerde = new JsonSerde<>(PBestUpdate.class);

        CoordinatorControl control = new CoordinatorControl();

        StreamsBuilder builder = new StreamsBuilder();

        // var globalStream = builder.stream(LOCAL_WEIGHTS_TOPIC, Consumed.with(Serdes.String(), Serdes.String()))
        //                         .process(() -> new CoordinatorProcessor(control));

        // =======================================================================================================
        
        KStream<String, String> local_weights_stream = builder.stream(
            LOCAL_WEIGHTS_TOPIC,
            Consumed.with(Serdes.String(), Serdes.String())
        );

        local_weights_stream.process(() -> new CoordinatorProcessor(control));

        // =======================================================================================================

        if(FULLY_INFORMED != "true") {

            KStream<String, String> pBestJsonStream = builder.stream(
                PBEST_WEIGHTS_TOPIC,
                Consumed.with(Serdes.String(), Serdes.String())
            )
            .peek((k, json) ->
                logger.log("New gBest from worker JSON: " + json)
            )
            .selectKey((k, v) -> "gBest");

            KTable<String, String> gBestTable = pBestJsonStream
                .groupByKey()
                .aggregate(
                    () -> null,               // initial aggregate = null (no gBest yet)
                    (key, newJson, aggJson) -> {
                        if (aggJson == null) return newJson;

                        try {
                            Map<String, Object> newMsg =
                                MAPPER.readValue(newJson, new TypeReference<Map<String, Object>>() {});
                            Map<String, Object> oldMsg =
                                MAPPER.readValue(aggJson, new TypeReference<Map<String, Object>>() {});

                            double newAcc = ((Number) newMsg.get("accuracy")).doubleValue();
                            double oldAcc = ((Number) oldMsg.get("accuracy")).doubleValue();
                            
                            // System.out.println("newJson: " + newJson);
                            // logger.log("I am running2");
                            return newAcc > oldAcc ? newJson : aggJson;
                        } catch (Exception e) {
                            e.printStackTrace();
                            return aggJson; // keep the old best if parsing fails
                        }
                    },
                    Materialized.<String, String, KeyValueStore<Bytes, byte[]>>as("gBestStore")
                        .withKeySerde(Serdes.String())
                        .withValueSerde(Serdes.String())
                        .withCachingDisabled()
            );

            gBestTable
                .toStream()
                .mapValues(json -> {
                    try {
                        // logger.log("I am running");
                        Map<String, Object> msg =
                            MAPPER.readValue(json, new TypeReference<Map<String, Object>>() {});

                        Map<String, Object> payload = new HashMap<>();
                        // reuse fields from the best pBest
                        payload.put("id_worker", msg.get("id_worker"));
                        payload.put("accuracy", msg.get("accuracy"));
                        payload.put("pBestMsgIndex", msg.get("pBestMsgIndex"));
                        // gBest weights are just pBest weights of the best particle
                        payload.put("w_gBest", msg.get("pBest"));

                        return MAPPER.writeValueAsString(payload);
                    } catch (Exception e) {
                        e.printStackTrace();
                        return null;
                    }
                })
                .filter((k, v) -> v != null)
                .peek((k, json) -> {
                    logger.log("New gBest JSON: " + json);
                    // System.out.println("[Coordinator] New gBest JSON: " + json);
                })
                .to(GLOBAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), Serdes.String()));
        }

        // =======================================================================================================

        Topology topology = builder.build();

        System.out.println("Coordinator topology:");
        // System.out.println(topology.describe());

        KafkaStreams streams = new KafkaStreams(topology, props);

        CountDownLatch latch = new CountDownLatch(1);

        streams.setUncaughtExceptionHandler((Thread t, Throwable e) -> {
            Throwable cause = e;
            // StreamsException wraps the real cause
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
        Thread controlThread = new Thread(() -> {
            try {
                while (!control.isStopRequested()) {
                    Thread.sleep(500); // poll every 500ms
                }
                System.out.println("[Coordinator] Stopping because desired accuracy was reached.");
                streams.close();
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
            } finally {
                latch.countDown();
            }
        }, "coordinator-control-thread");

        controlThread.setDaemon(true);
        controlThread.start();

        // ===============================================================================================================

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {     // executes when doing Ctrl + C, SIGINT -> JVM -> addShutdownHook
            System.out.println("[Coordinator] Shutting down KafkaStreams");
            streams.close();
            latch.countDown();
        }));

        try {
            streams.start();
            System.out.println("[Coordinator] KafkaStreams started.");
            latch.await(); 
        } catch (Throwable e) {
            System.out.println("[Coordinator] Error in KafkaStreams: " + e.getMessage());
            streams.close();
            latch.countDown();
        }

        System.out.println("[Coordinator] Exiting run()");

    }
}

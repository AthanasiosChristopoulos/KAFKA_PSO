package pso;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.*;
import org.apache.kafka.streams.kstream.Consumed;
import org.apache.kafka.streams.kstream.Produced;
import org.apache.kafka.clients.consumer.ConsumerConfig;

import org.apache.kafka.streams.KafkaStreams;

import java.util.Properties;

import java.util.concurrent.CountDownLatch;

import utils.*; 

public class Coordinator implements Runnable {

    private final String LOCAL_WEIGHTS_TOPIC;
    private final String GLOBAL_WEIGHTS_TOPIC;
    private final String RUN_ID;

    public Coordinator() {
        Config cfg = Config.get();
        this.RUN_ID = cfg.RUN_ID;
        this.LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
        this.GLOBAL_WEIGHTS_TOPIC = cfg.GLOBAL_WEIGHTS_TOPIC;
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

        CoordinatorControl control = new CoordinatorControl();

        StreamsBuilder builder = new StreamsBuilder();

        var globalStream = builder.stream(LOCAL_WEIGHTS_TOPIC, Consumed.with(Serdes.String(), Serdes.String()))
                                .process(() -> new CoordinatorProcessor(control));

        globalStream.to(GLOBAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), Serdes.String()));

        Topology topology = builder.build();

        System.out.println("Coordinator topology:");
        System.out.println(topology.describe());

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

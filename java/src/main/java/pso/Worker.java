
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
import org.apache.kafka.common.MetricName;
import org.apache.kafka.common.serialization.Serde;
import org.apache.kafka.common.Metric;
import org.apache.kafka.common.MetricName;

import org.apache.kafka.clients.producer.ProducerConfig;

import java.util.Properties;
import java.util.Map;
import utils.*;
import state.*;
import message.data_message.*; 
import message.weights_message.*; 

import java.util.concurrent.atomic.AtomicLong;

public class Worker implements Runnable {

    private final int workerId;
    
    private static Config cfg = Config.getInstance();
    private final String DATA_TOPIC = cfg.DATA_TOPIC;
    private final String PBEST_WEIGHTS_TOPIC = cfg.PBEST_WEIGHTS_TOPIC;
    private final String LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
    private final String GLOBAL_WEIGHTS_TOPIC = cfg.GLOBAL_WEIGHTS_TOPIC;
    private final String RUN_ID = cfg.RUN_ID;  
    private final boolean FULLY_INFORMED = cfg.FULLY_INFORMED;
    private final boolean DEBUG_KAFKA = cfg.DEBUG_KAFKA;
    private final boolean ENABLE_NEIGHBORHOODS = cfg.ENABLE_NEIGHBORHOODS;
    public final boolean INDEPENDENT_WORKER_DATA_PROCESSING = cfg.INDEPENDENT_WORKER_DATA_PROCESSING;

    private String stateStoreName;
    private String keyName;

    private final CoordinatorControl control;   // the coordinator is the one who finished when he has exhausted all the testing data
                                                // the he requestStop on the control and everything closes
    private long t0 = System.nanoTime();
    private final AtomicLong t1  = new AtomicLong(t0);

    private WorkerStatic ws;
    private CustomLogger logger;

    // =====================================================================================================

    public Worker(int workerId) {

        this.workerId = workerId;

        if(ENABLE_NEIGHBORHOODS == true  || FULLY_INFORMED == true) {
            stateStoreName = "pBestStore";
            keyName = "pBest" + workerId;
        } else {
            stateStoreName = "gBestStore";
            keyName = "gBest";
        }

        this.control = CoordinatorControl.getInstance();
        this.logger = CustomLogger.getWorkerInstance(workerId);

    }

    // =====================================================================================================

    @Override
    public void run() {
        try {
            runInternal();
        } catch (Throwable t) {
            System.err.println("[Worker " + workerId + "] FATAL in worker thread:");
            t.printStackTrace();
            CoordinatorControl.getInstance().requestStopFinal();
        }
    }
    // ==========================================================================================

    private void runInternal() throws Exception {

        this.ws = new WorkerStatic(workerId);
        System.out.println("[Worker " + workerId + "] with RUN_ID = " + RUN_ID + ", Thread: " + Thread.currentThread().getName());
        logger.log("[Worker " + workerId + "] with RUN_ID = " + RUN_ID + ", Thread: " + Thread.currentThread().getName());

        Properties props = new Properties();
        String applicationID;
        if (INDEPENDENT_WORKER_DATA_PROCESSING) {
            applicationID = "pso-worker-" + workerId + "_" + RUN_ID;
        } else {
            applicationID = "pso-worker-_" + RUN_ID;
        }

        props.put(StreamsConfig.APPLICATION_ID_CONFIG, applicationID);
        // props.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-worker-" + workerId + "_" + RUN_ID);     // different group Id, processing of the same data
        // props.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-worker-" + "_" + RUN_ID);        // same group Id, parallel processing
       
        props.put(StreamsConfig.STATE_DIR_CONFIG, "/tmp/kstreams-" + RUN_ID + "-worker-" + workerId);
        props.put(StreamsConfig.CLIENT_ID_CONFIG, "pso-worker-" + workerId + "-RUN-" + RUN_ID);
            // Kafka Streams uses this that client id as a prefix when naming its threads,

        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092"); // for now localhost, but this is the URL of the Kafka cluster
        props.put(org.apache.kafka.clients.consumer.ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        // props.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 0);
        // props.put(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, 1);
        // props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "2");
        props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1"); // 2 is pointless. The Global table consumer thread takes care of task 0
        props.put(StreamsConfig.producerPrefix(ProducerConfig.MAX_REQUEST_SIZE_CONFIG), 5 * 1024 * 1024); // 5 MB
        
        if(INDEPENDENT_WORKER_DATA_PROCESSING) {
            props.put(ConsumerConfig.MAX_POLL_INTERVAL_MS_CONFIG, 10 * 60 * 1000); // 10 minutes
            props.put(ConsumerConfig.SESSION_TIMEOUT_MS_CONFIG, 30000);
            props.put(ConsumerConfig.HEARTBEAT_INTERVAL_MS_CONFIG, 10000);
        }
        
        Serde<DataMessage> dataSerde = new DataMessageSerde();
        Serde<WeightsMessage> weightsSerde = new WeightsMessageSerde();

        StreamsBuilder builder = new StreamsBuilder();

        // Task 0 (of Global Streams) ===============================================================================
        // input stream 4 and input stream 7

        if(ENABLE_NEIGHBORHOODS == true || FULLY_INFORMED == true) {

            GlobalKTable<String, WeightsMessage> pBestTable = builder.globalTable(
                PBEST_WEIGHTS_TOPIC,    // messages from this are keyed differently for every worker
                Consumed.with(Serdes.String(), weightsSerde),
                Materialized.<String, WeightsMessage>as(Stores.inMemoryKeyValueStore(stateStoreName))
                    .withKeySerde(Serdes.String())
                    .withValueSerde(weightsSerde)
            );

        } else {
            
            GlobalKTable<String, WeightsMessage> gBestTable = builder.globalTable(
                GLOBAL_WEIGHTS_TOPIC,   // messages from this have the same key, "gBest",
                                        // we get only overwrites of gBest there is only one gBest
                Consumed.with(Serdes.String(), weightsSerde),
                Materialized.<String, WeightsMessage>as(Stores.inMemoryKeyValueStore(stateStoreName))
                    .withKeySerde(Serdes.String())
                    .withValueSerde(weightsSerde)
            );
        }

        // Task 1 ================================================================================================
        // input stream 1 and output stream 2_1 and stream 2_2
        
        KStream<String, DataMessage> rawDataStream = builder.stream(
            DATA_TOPIC,
            Consumed.with(Serdes.String(), dataSerde)
        );

        KStream<String, WeightsMessage> dataStream = rawDataStream
            .transform(() -> new WorkerTransformer(workerId, t0, t1, ws))
            .filter((k, v) -> v != null);

        KStream<String, WeightsMessage>[] branches = dataStream.branch(
            (key, value) -> keyName.equals(key),   // branch[0]: pBest / gBest updates
            (key, value) -> true                   // branch[1]: all others (weights)
        );

        branches[0].to(PBEST_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));
        branches[1].to(LOCAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));

        // =====================================================================================================
        // =====================================================================================================

        Topology topology = builder.build();

        KafkaStreams streams = new KafkaStreams(topology, props);

        streams.setUncaughtExceptionHandler((Thread t, Throwable e) -> {
            System.out.println("[Worker " + workerId + "] exception in Thread " + t.getName() + "requesting final stop.");
            e.printStackTrace();
            CoordinatorControl.getInstance().requestStopFinal();
            streams.close();
        });

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("[Worker " + workerId + "] Shutting down KafkaStreams");
            streams.close();
        }));

        streams.start();
        startMetricsLogger(streams); 

        System.out.println("[Worker " + workerId + "] started.");

        // if(workerId == 0 && DEBUG_KAFKA == true) {
        if(DEBUG_KAFKA == true) {
            System.out.println("[Worker " + workerId + "] Topology:\n" + topology.describe());

            try { 
                Thread.sleep(1500); 
            } catch (InterruptedException ignored) {
                System.out.println("Sleep failed");
            }

            for (ThreadMetadata tm : streams.localThreadsMetadata()) {
                System.out.println("Thread: " + tm.threadName() + " state=" + tm.threadState());

                for (TaskMetadata task : tm.activeTasks()) {
                    System.out.println("  ACTIVE Task: " + task.taskId() + " partitions=" + task.topicPartitions());
                }

            }
        }

        while (!control.isStopRequested(workerId)) {
            Thread.sleep(50);  
        }
        streams.close();

        double seconds = (t1.get() - t0) / 1_000_000_000.0;     // t1 is updated at WorkerTransformer every time a new buffer has been processed
        System.out.printf("[Worker %d] Elapsed time: %.3f seconds, exiting run()%n", workerId, seconds);

    }

    //====================================================================================================================

    private void startMetricsLogger(KafkaStreams streams) {
        final Map<MetricName, ? extends Metric> metrics = streams.metrics();

        Thread t = new Thread(() -> {
            while (!control.isStopRequested(workerId)) {
                try { Thread.sleep(1000); } catch (InterruptedException ignored) {}

                double reqLatAvg = Double.NaN;
                double outByteRate = Double.NaN;
                double sendRate = Double.NaN;
                double bufferWaitTotal = Double.NaN;

                for (Map.Entry<MetricName, ? extends Metric> e : metrics.entrySet()) {
                    MetricName name = e.getKey();
                    if (!"producer-metrics".equals(name.group())) continue;

                    String n = name.name();
                    Object v = e.getValue().metricValue();
                    if (!(v instanceof Number)) continue;

                    double dv = ((Number) v).doubleValue();

                    switch (n) {
                        case "request-latency-avg": reqLatAvg = dv; break;
                        case "outgoing-byte-rate": outByteRate = dv; break;
                        case "record-send-rate": sendRate = dv; break;
                        case "bufferpool-wait-time-total": bufferWaitTotal = dv; break;
                        default: break;
                    }
                }

                logger.log("producer-metrics: " +
                    "request-latency-avg = " + reqLatAvg +
                    ", outgoing-byte-rate = " + outByteRate +
                    ", record-send-rate = " + sendRate +
                    ", bufferpool-wait-time-total = " + bufferWaitTotal
                );
            }
        });

        t.setDaemon(true);
        t.setName("metrics-logger-worker-" + workerId);
        t.start();
    }

}


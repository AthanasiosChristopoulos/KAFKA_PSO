
package pso;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.*;
import org.apache.kafka.streams.kstream.Consumed;
import org.apache.kafka.streams.kstream.Produced;
import org.apache.kafka.streams.kstream.Materialized;
import org.apache.kafka.streams.kstream.KTable;
import org.apache.kafka.streams.kstream.GlobalKTable;
import org.apache.kafka.streams.kstream.Grouped;
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
import java.util.Set;
import java.util.HashSet;
import org.apache.kafka.clients.producer.ProducerConfig;

import java.util.Properties;
import java.util.Map;
import utils.*;
import state.*;
import experimentation.*;
import message.data_message.*; 
import message.weights_message.*; 

import java.util.concurrent.atomic.AtomicLong;

public class Worker implements Runnable {

    private final int workerId;
    
    private static Config cfg = Config.getInstance();
    private final String DATA_TOPIC = cfg.DATA_TOPIC;
    private final String PBEST_WEIGHTS_TOPIC = cfg.PBEST_WEIGHTS_TOPIC;
    private final String LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
    private final String GPEST_WEIGHTS_TOPIC = cfg.GPEST_WEIGHTS_TOPIC;
    private final String RUN_ID = cfg.RUN_ID;  
    private final boolean FULLY_INFORMED = cfg.FULLY_INFORMED;
    private final boolean DEBUG_KAFKA = cfg.DEBUG_KAFKA;
    private final boolean ENABLE_NEIGHBORHOODS = cfg.ENABLE_NEIGHBORHOODS;
    public final boolean INDEPENDENT_DATA_PROCESSING = cfg.INDEPENDENT_DATA_PROCESSING;

    private String stateStoreName;
    private String keyName;

    private final CoordinatorControl control;   // the coordinator is the one who finished when he has exhausted all the testing data
                                                // the he requestStop on the control and everything closes
    private long t0 = System.nanoTime();
    private final AtomicLong t1  = new AtomicLong(t0);

    private WorkerStatic ws;
    private CustomLogger logger;

    private final MetricsCollector collector; 
    private boolean KAFKA_METRICS_ENABLED = false;
    
    // =====================================================================================================

    public Worker(int workerId, MetricsCollector collector) {

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

        this.collector = collector;
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
        if (INDEPENDENT_DATA_PROCESSING) {
            applicationID = "pso-worker-" + workerId + "_" + RUN_ID;
        } else {
            applicationID = "pso-worker-_" + RUN_ID;
        }

        props.put(StreamsConfig.APPLICATION_ID_CONFIG, applicationID);
        // props.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-worker-" + workerId + "_" + RUN_ID);     // different group Id, processing of the same data
        // props.put(StreamsConfig.APPLICATION_ID_CONFIG, "pso-worker-" + "_" + RUN_ID);        // same group Id, parallel processing
       
        props.put(StreamsConfig.STATE_DIR_CONFIG, cfg.KAFKA_TMP_DIR + "-" + RUN_ID + "-worker-" + workerId);
        props.put(StreamsConfig.CLIENT_ID_CONFIG, "pso-worker-" + workerId + "-RUN-" + RUN_ID);
            // Kafka Streams uses this that client id as a prefix when naming its threads,

        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, cfg.KAFKA_HOST); // for now localhost, but this is the URL of the Kafka cluster
        props.put(org.apache.kafka.clients.consumer.ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        // props.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 0);
        // props.put(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, 1);
        // props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "2");
        props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1"); // 2 is pointless. The Global table consumer thread takes care of task 0
        props.put(StreamsConfig.producerPrefix(ProducerConfig.MAX_REQUEST_SIZE_CONFIG), 5 * 1024 * 1024); // 5 MB
        props.put(StreamsConfig.producerPrefix(ProducerConfig.LINGER_MS_CONFIG), 0);
        // props.put("statestore.cache.max.bytes", 50 * 1024 * 1024L); // e.g. 50MB
        // props.put("statestore.cache.max.bytes", 0L);
        // props.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 0); // e.g. flush every 100ms

        if(INDEPENDENT_DATA_PROCESSING) {
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
                GPEST_WEIGHTS_TOPIC,   // messages from this have the same key, "gBest",
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

        if(false && cfg.FILTER_ENABLED) {

            KStream<String, WeightsMessage> pBestStream = branches[0]
                .peek((k,v) -> System.out.println("PBEST IN  key = " + k + " msgIndex = " + v.msgIndex));

            KTable<String, WeightsMessage> pBestLatest = pBestStream
                .groupByKey(Grouped.with(Serdes.String(), weightsSerde))
                .reduce((oldV, newV) -> newV, Materialized.as("pbest-latest-store"));

            pBestLatest.toStream()
                .peek((k,v) -> System.out.println("PBEST OUT key = " + k + " msgIndex = " + v.msgIndex))
                .to(PBEST_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));

            branches[1].to(LOCAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));

        } else {

            branches[0].to(PBEST_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));
            branches[1].to(LOCAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));
        }

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
        if(KAFKA_METRICS_ENABLED) {
            startMetricsLogger(streams);
        } 
        // dumpProducerMetricNamesOnce(streams);
        // startBatchingProofLogger(streams);
        
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

        double seconds = (t1.get() - t0) / 1_000_000_000.0;    
        System.out.printf("[Worker %d] Elapsed time: %.3f seconds, exiting run()%n", workerId, seconds);
        float bestAcc = ws.stats.getBestAccuracy();
        float bestLoss = ws.stats.getPBestLoss();   

        if (collector != null) {
            if(logger.isEnabled(0)) logger.log(
                "ws.TOTAL_MESSAGES_SENT: " + ws.TOTAL_MESSAGES_SENT +
                "ws.TOTAL_MESSAGES_SENT_PBEST: " + ws.TOTAL_MESSAGES_SENT_PBEST + 
                "ws.TOTAL_MESSAGES_SENT_CURRENT_WEIGHTS:" + ws.TOTAL_MESSAGES_SENT_CURRENT_WEIGHTS);

            collector.reportWorkerDone(new WorkerMetrics(workerId, seconds, bestAcc, bestLoss, 
                ws.TOTAL_MESSAGES_SENT, ws.TOTAL_MESSAGES_SENT_PBEST,ws.TOTAL_MESSAGES_SENT_CURRENT_WEIGHTS,
                ws.TOTAL_BYTES_SENT));
        }
    }

    //====================================================================================================================
    //====================================================================================================================
    //====================================================================================================================
    //====================================================================================================================

    private void startMetricsLogger(KafkaStreams streams) {
        final Map<MetricName, ? extends Metric> metrics = streams.metrics();

        Thread t = new Thread(() -> {
            while (!control.isStopRequested(workerId)) {
                try { Thread.sleep(1000); } catch (InterruptedException ignored) {}

                // =========================
                // Producer metrics (filter affects PBEST producing -> these may change)
                // =========================
                double prodReqLatAvg = Double.NaN;
                double prodOutByteRate = Double.NaN;
                double prodRecordSendRate = Double.NaN;
                double prodBufferWaitTotal = Double.NaN; // "time lost" due to backpressure (if any)

                // =========================
                // Consumer metrics for GlobalKTable consumer only (best-effort by client-id tag)
                // =========================
                Agg gFetchLatencyAvg = new Agg(AggMode.AVG);
                Agg gFetchRate = new Agg(AggMode.SUM);
                Agg gBytesConsumedRate = new Agg(AggMode.SUM);
                Agg gRecordsConsumedRate = new Agg(AggMode.SUM);
                Agg gPollLatencyAvg = new Agg(AggMode.AVG);
                Agg gRecordsLagMax = new Agg(AggMode.MAX);  // may not exist

                for (Map.Entry<MetricName, ? extends Metric> e : metrics.entrySet()) {
                    MetricName mn = e.getKey();

                    Object vObj = e.getValue().metricValue();
                    if (!(vObj instanceof Number)) continue;
                    double v = ((Number) vObj).doubleValue();

                    final String group = mn.group();
                    final String name = mn.name();

                    // -------- Producer metrics (global for this KafkaStreams instance) --------
                    if ("producer-metrics".equals(group)) {
                        switch (name) {
                            case "request-latency-avg": prodReqLatAvg = v; break;
                            case "outgoing-byte-rate": prodOutByteRate = v; break;
                            case "record-send-rate": prodRecordSendRate = v; break;
                            case "bufferpool-wait-time-total": {
                                prodBufferWaitTotal = v; 
                                if(v != 0) {
                                    System.out.println("CONGESTION");
                                }
                                break;
                            }
                            default: break;
                        }
                        continue;
                    }

                    // -------- Consumer metrics: only keep the GlobalKTable consumer(s) --------
                    if ("consumer-metrics".equals(group)) {
                        // Filter by client-id tag to exclude DATA_TOPIC consumer and keep global-table consumer.
                        // This is best-effort; tag keys/values vary by Kafka version.
                        if (!isGlobalTableConsumer(mn)) continue;

                        switch (name) {
                            case "fetch-latency-avg": gFetchLatencyAvg.add(v); break;
                            case "fetch-rate": gFetchRate.add(v); break;
                            case "bytes-consumed-rate": gBytesConsumedRate.add(v); break;
                            case "records-consumed-rate": gRecordsConsumedRate.add(v); break;
                            case "poll-latency-avg": gPollLatencyAvg.add(v); break;
                            case "records-lag-max": gRecordsLagMax.add(v); break;
                            default: break;
                        }
                    }
                }
                
                logger.log(
                    "[metrics-filter-relevant]" + "\n" + 
                    "producers: " +
                    "request-latency-avg=" + fmt(prodReqLatAvg) +
                    ", outgoing-byte-rate=" + fmt(prodOutByteRate) +
                    ", record-send-rate=" + fmt(prodRecordSendRate) +
                    ", bufferpool-wait-time-total=" + fmt(prodBufferWaitTotal) + "\n" +
                    "global-consumer: " +
                    "fetch-latency-avg=" + fmt(gFetchLatencyAvg.value()) +
                    ", fetch-rate=" + fmt(gFetchRate.value()) +
                    ", bytes-consumed-rate=" + fmt(gBytesConsumedRate.value()) +
                    ", records-consumed-rate=" + fmt(gRecordsConsumedRate.value()) +
                    ", poll-latency-avg=" + fmt(gPollLatencyAvg.value()) +
                    ", records-lag-max=" + fmt(gRecordsLagMax.value())
                );
            }
        });

        t.setDaemon(true);
        t.setName("metrics-logger-worker-" + workerId);
        t.start();
    }

    // ====================================================================================

    private static boolean isGlobalTableConsumer(MetricName mn) {
        Map<String, String> tags = mn.tags();
        if (tags == null) return false;

        String clientId = tags.get("client-id");
        if (clientId == null) return false;

        String s = clientId.toLowerCase(java.util.Locale.ROOT);

        if(s.contains("global") || s.contains("globalstreamthread")) {
            // System.out.println("AAAAA: " + s);
            return true;

        }
        return false;
    }

    // ====================================================================================

    private static String fmt(double v) {
        if (Double.isNaN(v)) return "NaN";
        return String.format(java.util.Locale.ROOT, "%.3f", v);
    }

    private enum AggMode { SUM, AVG, MAX }

    private static final class Agg {
        private final AggMode mode;
        private double sum = 0.0;
        private long count = 0;
        private double max = Double.NEGATIVE_INFINITY;

        Agg(AggMode mode) { this.mode = mode; }

        void add(double v) {
            switch (mode) {
                case SUM:
                case AVG:
                    sum += v;
                    count++;
                    break;
                case MAX:
                    if (v > max) max = v;
                    count++;
                    break;
            }
        }

        double value() {
            if (count == 0) return Double.NaN;
            switch (mode) {
                case SUM: return sum;
                case AVG: return sum / count;
                case MAX: return max;
                default: return Double.NaN;
            }
        }
    }

    // ====================================================================================

    private void startBatchingProofLogger(KafkaStreams streams) {
        final Map<MetricName, ? extends Metric> metrics = streams.metrics();

        Thread t = new Thread(() -> {
            while (!control.isStopRequested(workerId)) {
                try { Thread.sleep(1000); } catch (InterruptedException ignored) {}

                double batchSizeAvg = Double.NaN;
                double batchSizeMax = Double.NaN;
                double recordsPerRequestAvg = Double.NaN;
                double recordQueueTimeAvg = Double.NaN;
                double requestRate = Double.NaN;
                double recordSendRate = Double.NaN;

                for (Map.Entry<MetricName, ? extends Metric> e : metrics.entrySet()) {
                    MetricName mn = e.getKey();
                    if (!"producer-metrics".equals(mn.group())) continue;

                    Object vObj = e.getValue().metricValue();
                    if (!(vObj instanceof Number)) continue;
                    double v = ((Number) vObj).doubleValue();

                    switch (mn.name()) {
                        case "batch-size-avg": batchSizeAvg = v; break;
                        case "batch-size-max": batchSizeMax = v; break;
                        case "records-per-request-avg": recordsPerRequestAvg = v; break;
                        case "record-queue-time-avg": recordQueueTimeAvg = v; break;
                        case "request-rate": requestRate = v; break;
                        case "record-send-rate": recordSendRate = v; break;
                        default: break;
                    }
                }

                // Derived proof signals
                double recordsPerRequestFromRates = Double.NaN;
                if (!Double.isNaN(recordSendRate) && !Double.isNaN(requestRate) && requestRate > 0.0) {
                    recordsPerRequestFromRates = recordSendRate / requestRate;
                }

                // Proof conditions (human-readable)
                String proof1 = (!Double.isNaN(recordsPerRequestAvg) && recordsPerRequestAvg > 1.05)
                    ? "BATCHING_PROVEN(records-per-request-avg>1)"
                    : "records-per-request-avg not proving";

                String proof2 = (!Double.isNaN(recordsPerRequestFromRates) && recordsPerRequestFromRates > 1.05)
                    ? "BATCHING_LIKELY(record-send-rate/request-rate>1)"
                    : "rate-ratio not proving";

                logger.log(
                    "[batching-proof] " +
                    "batch-size-avg=" + fmt(batchSizeAvg) +
                    ", batch-size-max=" + fmt(batchSizeMax) +
                    ", records-per-request-avg=" + fmt(recordsPerRequestAvg) +
                    ", record-queue-time-avg=" + fmt(recordQueueTimeAvg) +
                    ", request-rate=" + fmt(requestRate) +
                    ", record-send-rate=" + fmt(recordSendRate) +
                    ", send/request=" + fmt(recordsPerRequestFromRates) +
                    " | " + proof1 + " | " + proof2
                );
            }
        });

        t.setDaemon(true);
        t.setName("batching-proof-worker-" + workerId);
        t.start();
    }

    // ====================================================================================

    private void dumpProducerMetricNamesOnce(KafkaStreams streams) {
        Set<String> names = new HashSet<>();
        for (MetricName mn : streams.metrics().keySet()) {
            if ("producer-metrics".equals(mn.group())) {
                names.add(mn.name());
            }
        }
        System.out.println("[Worker " + workerId + "] producer-metrics available: " + names);
        logger.log("[Worker " + workerId + "] producer-metrics available: " + names);
    }

    // private void startMetricsLogger(KafkaStreams streams) {
    //     final Map<MetricName, ? extends Metric> metrics = streams.metrics();

    //     Thread t = new Thread(() -> {
    //         while (!control.isStopRequested(workerId)) {
    //             try { Thread.sleep(1000); } catch (InterruptedException ignored) {}

    //             double reqLatAvg = Double.NaN;
    //             double outByteRate = Double.NaN;
    //             double sendRate = Double.NaN;
    //             double bufferWaitTotal = Double.NaN;

    //             for (Map.Entry<MetricName, ? extends Metric> e : metrics.entrySet()) {
    //                 MetricName name = e.getKey();
    //                 if (!"producer-metrics".equals(name.group())) continue;

    //                 String n = name.name();
    //                 Object v = e.getValue().metricValue();
    //                 if (!(v instanceof Number)) continue;

    //                 double dv = ((Number) v).doubleValue();

    //                 switch (n) {
    //                     case "request-latency-avg": reqLatAvg = dv; break;
    //                     case "outgoing-byte-rate": outByteRate = dv; break;
    //                     case "record-send-rate": sendRate = dv; break;
    //                     case "bufferpool-wait-time-total": bufferWaitTotal = dv; break;
    //                     default: break;
    //                 }
    //             }

    //             logger.log("producer-metrics: " +
    //                 "request-latency-avg = " + reqLatAvg +
    //                 ", outgoing-byte-rate = " + outByteRate +
    //                 ", record-send-rate = " + sendRate +
    //                 ", bufferpool-wait-time-total = " + bufferWaitTotal
    //             );
    //         }
    //     });

    //     t.setDaemon(true);
    //     t.setName("metrics-logger-worker-" + workerId);
    //     t.start();
    // }

}


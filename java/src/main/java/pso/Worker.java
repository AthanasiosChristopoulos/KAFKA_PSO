
    package pso;

    import org.apache.kafka.common.serialization.Serdes;
    import org.apache.kafka.common.serialization.Serde;
    import org.apache.kafka.clients.producer.ProducerConfig;
    import org.apache.kafka.clients.consumer.ConsumerConfig;
    import org.apache.kafka.streams.*;
    import org.apache.kafka.streams.kstream.Consumed;
    import org.apache.kafka.streams.kstream.Produced;
    import org.apache.kafka.streams.kstream.Materialized;
    import org.apache.kafka.streams.kstream.GlobalKTable;
    import org.apache.kafka.streams.kstream.KStream;
    import org.apache.kafka.streams.state.Stores;
    import org.apache.kafka.streams.processor.ThreadMetadata;
    import org.apache.kafka.streams.processor.TaskMetadata;

    import java.util.Set;
    import java.util.HashSet;
    import java.util.Properties;
    import java.util.concurrent.atomic.AtomicLong;

    import utils.*;
    import state.*;
    import experimentation.*;
    import message.data_message.*; 
    import message.weights_message.*; 

    public class Worker implements Runnable {

        private final int workerId;
        
        private static Config cfg = Config.getInstance();
        private final String DATA_TOPIC = cfg.DATA_TOPIC;
        private final String PBEST_WEIGHTS_TOPIC = cfg.PBEST_WEIGHTS_TOPIC;
        private final String LOCAL_WEIGHTS_TOPIC = cfg.LOCAL_WEIGHTS_TOPIC;
        private final String GBEST_WEIGHTS_TOPIC = cfg.GBEST_WEIGHTS_TOPIC;
        private final String RUN_ID = cfg.RUN_ID;  
        private final boolean FULLY_INFORMED = cfg.FULLY_INFORMED;
        private final boolean DEBUG_KAFKA = cfg.DEBUG_KAFKA;
        public final boolean DATASET_PARTITIONING = cfg.DATASET_PARTITIONING;

        private String stateStoreName;
        private String keyName;

        private final CoordinatorControl control;  
                                        
        private long t0 = System.nanoTime();
        private final AtomicLong t_actually_started = new AtomicLong(t0);
        private final AtomicLong t1  = new AtomicLong(t0);

        private WorkerStatic ws;
        private CustomLogger logger;

        private final MetricsCollector collector; 
        
        // =========================================================================
        public Worker(int workerId, MetricsCollector collector) {

            this.workerId = workerId;

            if(FULLY_INFORMED == true || cfg.ENABLE_NEIGHBORHOODS) {
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

        // ===========================================================================
        
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

            // verifyKafkaConnectivityOrThrow();
            
            if(cfg.EXPERIMENTATION_MODE.equals("MONITORING_ITERATIONS")) {  // Wait specifically in this cases to make sure coordinator has loaded the entire test 
                try{
                    Thread.sleep(4000);
                } catch(Exception e) {
                    System.out.println("Sleeping didnt work");
                    e.printStackTrace();
                }
            }
            
            t_actually_started.set(System.nanoTime());

            this.ws = new WorkerStatic(workerId);
            System.out.println("[Worker " + workerId + "] with RUN_ID = " + RUN_ID + ", Thread: " + Thread.currentThread().getName());
            logger.log("[Worker " + workerId + "] with RUN_ID = " + RUN_ID + ", Thread: " + Thread.currentThread().getName());

            Properties props = new Properties();
            String applicationID;
            if (!DATASET_PARTITIONING) {
                applicationID = "pso-worker-" + workerId + "_" + RUN_ID;
            } else {
                applicationID = "pso-worker-_" + RUN_ID;
            }

            final Set<Integer> pBestRecipientWorkerIds = computePBestRecipientsForThisWorker();

            logger.log("[ROUTING-INIT] sender = " + workerId + " recipients = " + pBestRecipientWorkerIds);

            props.put(StreamsConfig.APPLICATION_ID_CONFIG, applicationID);
            props.put(StreamsConfig.STATE_DIR_CONFIG, cfg.KAFKA_TMP_DIR + "-" + RUN_ID + "-worker-" + workerId);
            props.put(StreamsConfig.CLIENT_ID_CONFIG, "pso-worker-" + workerId + "-RUN-" + RUN_ID);
                // Kafka Streams uses this that client id as a prefix when naming its threads
            System.out.println("cfg.KAFKA_HOST: " + cfg.KAFKA_HOST);
            props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, cfg.KAFKA_HOST);  // this is the URL of the Kafka cluster. Kafka Streams connects to Kafka using this network address
            props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
            props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, "1");    // This is number of kafka streams threads. 2 is pointless. The Global table consumer thread takes care of task 0
            props.put(StreamsConfig.producerPrefix(ProducerConfig.MAX_REQUEST_SIZE_CONFIG), 5 * 1024 * 1024); // 5 MB
            props.put(StreamsConfig.producerPrefix(ProducerConfig.LINGER_MS_CONFIG), 0);

            if(!DATASET_PARTITIONING) {
                props.put(ConsumerConfig.MAX_POLL_INTERVAL_MS_CONFIG, 10 * 60 * 1000); // 10 minutes
                props.put(ConsumerConfig.SESSION_TIMEOUT_MS_CONFIG, 30000);
                props.put(ConsumerConfig.HEARTBEAT_INTERVAL_MS_CONFIG, 10000);
            }
            
            Serde<DataMessage> dataSerde = new DataMessageSerde();
            Serde<WeightsMessage> weightsSerde = new WeightsMessageSerde();

            StreamsBuilder builder = new StreamsBuilder();

            // Task 0 (of Global Streams) ===============================================================================
            // input stream 4 and input stream 7

            if(FULLY_INFORMED == true || cfg.ENABLE_NEIGHBORHOODS) {

                if(cfg.ENABLE_NEIGHBORHOODS) {
                    GlobalKTable<String, WeightsMessage> pBestTable = builder.globalTable(
                        "PBEST-WORKER-" + workerId,    // messages from this are keyed differently for every worker
                        Consumed.with(Serdes.String(), weightsSerde),
                        Materialized.<String, WeightsMessage>as(Stores.inMemoryKeyValueStore(stateStoreName))
                            .withKeySerde(Serdes.String())
                            .withValueSerde(weightsSerde)
                    );
                } else {
                    GlobalKTable<String, WeightsMessage> pBestTable = builder.globalTable(
                        PBEST_WEIGHTS_TOPIC,    // messages from this are keyed differently for every worker
                        Consumed.with(Serdes.String(), weightsSerde),
                        Materialized.<String, WeightsMessage>as(Stores.inMemoryKeyValueStore(stateStoreName))
                            .withKeySerde(Serdes.String())
                            .withValueSerde(weightsSerde)
                    );
                }

            } else {
                
                GlobalKTable<String, WeightsMessage> gBestTable = builder.globalTable(
                    GBEST_WEIGHTS_TOPIC,   // messages from this have the same key, "gBest",
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
                .transform(() -> new WorkerTransformer(workerId, t0, t_actually_started, t1, ws))
                .filter((k, v) -> v != null);

            KStream<String, WeightsMessage>[] branches = dataStream.branch(
                (key, value) -> keyName.equals(key),   // branch[0]: pBest / gBest updates
                (key, value) -> true                   // branch[1]: all others (weights)
            );

            if (cfg.ENABLE_NEIGHBORHOODS) {
                for (int recipientWorkerId : pBestRecipientWorkerIds) {
                    final int targetWorkerId = recipientWorkerId;
                    final String workerPBestTopic = "PBEST-WORKER-" + targetWorkerId;

                    logger.log(
                        "[ROUTING-SETUP] sender=" + workerId +
                        " -> receiver=" + targetWorkerId +
                        " | topic=" + workerPBestTopic
                    );

                    branches[0]
                        .filter((key, value) -> value != null)
                        .peek((key, value) -> logger.log(
                            "[ROUTING] sender=" + workerId +
                            " -> receiver=" + targetWorkerId +
                            " | topic=" + workerPBestTopic +
                            " | msgWorkerId=" + value.workerId
                        ))
                        .to(workerPBestTopic, Produced.with(Serdes.String(), weightsSerde));
                }
            } else {
                branches[0].to(PBEST_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde));
            }
            // monitoring / current weights branch
            branches[1].to(LOCAL_WEIGHTS_TOPIC, Produced.with(Serdes.String(), weightsSerde)); 


            // ====================================================================================
            // ====================================================================================
            Topology topology = builder.build();

            KafkaStreams streams = new KafkaStreams(topology, props);
            streams.setUncaughtExceptionHandler((Thread t, Throwable e) -> {
                System.out.println("[Worker " + workerId + "] exception in Thread " + t.getName() + "requesting final stop.");
                e.printStackTrace();
                CoordinatorControl.getInstance().requestStopFinal();
                streams.close();
            });

            // Shutdown hook for Ctrl + C =================================================================
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                System.out.println("[Worker " + workerId + "] Shutting down KafkaStreams");
                streams.close();
            }));

            streams.start();
            t0 = System.nanoTime();
            
            System.out.println("[Worker " + workerId + "] started.");

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
            double starting_delay = (t_actually_started.get() - t0) / 1_000_000_000.0;    
            System.out.printf("[Worker %d] Elapsed time: %.3f seconds, starting delay: %.3f, exiting run()%n", workerId, seconds, starting_delay);
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

        // ======================================================================================================
        // Neighborhood implementation: =========================================================================
        // ======================================================================================================

        private Set<Integer> computePBestRecipientsForThisWorker() {
            Set<Integer> recipients = new HashSet<>();

            if (!cfg.ENABLE_NEIGHBORHOODS) {
                return recipients;
            }
            
            // Neighborhood mode:
            // recipient r should receive sender s iff s is in neighborhood(r)
            if (cfg.ENABLE_NEIGHBORHOODS) {
                int ringRadius = Math.max(0, cfg.NEIGHBORHOOD_SIZE / 2);

                for (int recipientWorkerId = 0; recipientWorkerId < cfg.N_WORKERS; recipientWorkerId++) {
                    int[] recipientNeighbors = computeNeighborIds(
                        recipientWorkerId,
                        cfg.N_WORKERS,
                        ringRadius,
                        cfg.INCLUDE_SELF,
                        cfg.NEIGHBORHOOD_TOPOLOGY
                    );

                    for (int neighborId : recipientNeighbors) {
                        if (neighborId == this.workerId) {
                            recipients.add(recipientWorkerId);
                            break;
                        }
                    }
                }
                return recipients;
            }

            // Fallback
            for (int i = 0; i < cfg.N_WORKERS; i++) {
                recipients.add(i);
            }
            return recipients;
        }

        // ================================================================================================

        private static int[] computeNeighborIds(int workerId, int nWorkers, int ringRadius, boolean includeSelf,String topology) {
            if (nWorkers <= 0) return new int[0];

            switch (topology) {
                case "square":
                    int minSquare = includeSelf ? 5 : 4;
                    if (nWorkers < minSquare) {
                        return allWorkerIds(workerId, nWorkers, includeSelf);
                    }
                    return computeSquareNeighborIds(workerId, nWorkers, includeSelf);

                case "ring":
                default:
                    int maxPossible = includeSelf ? nWorkers : (nWorkers - 1);
                    int requested = includeSelf ? (2 * ringRadius + 1) : (2 * ringRadius);
                    if (requested > maxPossible || nWorkers < cfg.NEIGHBORHOOD_SIZE) {
                        return allWorkerIds(workerId, nWorkers, includeSelf);
                    }
                    return computeRingNeighborIds(workerId, nWorkers, ringRadius, includeSelf);
            }
        }

        // ===============================================================================================

        private static int[] computeRingNeighborIds(int workerId, int nWorkers, int radius, boolean includeSelf) {
            
            if (radius <= 0) return includeSelf ? new int[]{ workerId } : new int[0];

            // Ensure we don't request more unique neighbors than exist
            radius = Math.min(radius, (nWorkers - 1) / 2);

            int size = includeSelf ? (2 * radius + 1) : (2 * radius);
            int[] ids = new int[size];
            int idx = 0;

            if (includeSelf) ids[idx++] = workerId;

            for (int d = 1; d <= radius; d++) {
                int left  = Math.floorMod(workerId - d, nWorkers);
                int right = Math.floorMod(workerId + d, nWorkers);
                ids[idx++] = left;
                ids[idx++] = right;
            }

            return ids;
        }

        // ===================================================================================

        private static int[] computeSquareNeighborIds(int workerId, int nWorkers, boolean includeSelf) {
            // each node talks to its 4 von-Neumann neighbors (up/down/left/right).

            int rows = (int) Math.floor(Math.sqrt(nWorkers));   // √N_WORKERS × √N_WORKERS torus grid (2D)
            int cols = rows;
            
            // If not a perfect square, degrade gracefully to a rectangle
            if (rows * cols != nWorkers) {
                cols = (int) Math.ceil((double) nWorkers / rows); // Now rows*cols may exceed nWorkers;
            }

            // This converts a linear index (workerId) into 2D grid coordinates.
            int row = workerId / cols;
            int col = workerId % cols;

            // these should translate into worker IDs - if they dont we need to mod with nWorkers:
            int up    = Math.floorMod(row - 1, rows) * cols + col;
            int down  = Math.floorMod(row + 1, rows) * cols + col;
            int left  = row * cols + Math.floorMod(col - 1, cols);
            int right = row * cols + Math.floorMod(col + 1, cols);

            up = Math.floorMod(up, nWorkers);   // Map back into [0, nWorkers) in case rows*cols > nWorkers
            down = Math.floorMod(down, nWorkers);
            left = Math.floorMod(left, nWorkers);
            right = Math.floorMod(right, nWorkers);

            if (includeSelf) {
                return new int[]{ workerId, up, down, left, right };
            } else {
                return new int[]{ up, down, left, right };
            }
        }

        // If N_WORKERS = 16, rows = 4, cols = 4
        // maps workerId to the grid, int row = workerId / cols; int col = workerId % cols;
        // Row 0:   0   1   2   3
        // Row 1:   4   5   6   7
        // Row 2:   8   9  10  11
        // Row 3:  12  13  14  15
        // id = r * cols + c
        // stuff like Math.floorMod(row - 1, rows) is used to wrap around the grid so as to:
            // neighbors for 0 => {12, 4, 3, 1}

        // If N_WORKERS = 10, do a rectangle, rows = 3, cols = 4  
        // r=0:  0  1  2  3
        // r=1:  4  5  6  7
        // r=2:  8  9 10 11 
        // For an id of 10, 11, then it automatically becomes a 0, 1 respectively

        // ============================================================================================
        
        private static int[] allWorkerIds(int workerId, int nWorkers, boolean includeSelf) {
            if (nWorkers <= 0) return new int[0];

            if (includeSelf) {
                int[] ids = new int[nWorkers];
                for (int i = 0; i < nWorkers; i++) ids[i] = i;  // 0 ... nWorkers - 1
                return ids;
            } else {    // here we need to explicitly exclude self. this is why nWorkers - 1
                if (nWorkers == 1) return new int[0];
                int[] ids = new int[nWorkers - 1];
                int idx = 0;
                for (int i = 0; i < nWorkers; i++) {
                    if (i == workerId) continue;
                    ids[idx++] = i;
                }
                return ids;
            }
        }

    }


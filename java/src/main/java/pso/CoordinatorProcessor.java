package pso;

import org.apache.kafka.streams.processor.api.ContextualProcessor;
import org.apache.kafka.streams.processor.api.Record;
import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.streams.processor.api.ProcessorContext;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.factory.Nd4j;

import dl4j_models.Dl4jParamUtils;
import dl4j_models.PsoModel;

import java.io.File;
import java.io.IOException;

import org.apache.kafka.streams.processor.api.Processor;
import org.apache.kafka.streams.processor.api.ProcessorContext;
import org.apache.kafka.streams.processor.api.Record;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.io.BufferedWriter;

import java.util.*;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.common.serialization.StringDeserializer;
import org.apache.kafka.streams.state.ValueAndTimestamp;

import java.time.Duration;

import java.util.concurrent.atomic.AtomicInteger;

import utils.*;
import state.*;
import message.data_message.*; 
import message.weights_message.*; 

import org.apache.kafka.common.TopicPartition;

public class CoordinatorProcessor implements Processor<String, WeightsMessage, String, WeightsMessage> {
    private ProcessorContext<String, WeightsMessage> context;

    private final Map<String, float[]> weightsBuffer = new HashMap<>(); // this should be a dictionary of N_WORKER unique "id_worker" keys

    private final PsoModel globalModel; // x_g , current model
    private final PsoModel bestGlobalModel; 
    private PsoModel preTrainedModel;

    private float accuracy = -1f;    
    private float loss = 10000f;
    private int nSamples = 0;
    private int nCorrect = 0;

    private float bestLoss = 10000f;

    private final BatchPrediction globalPredictor;

    private static Config cfg = Config.getInstance();
    private final int N_WORKERS = cfg.N_WORKERS;
    private final String TEST_TOPIC = cfg.TEST_TOPIC;
    private final int TEST_SIZE = cfg.TEST_SIZE;
    private final float DESIRED_ACCURACY = cfg.DESIRED_ACCURACY;
    private final String RUN_ID = cfg.RUN_ID;   
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT; 
    private final String SAVE_MODEL_NAME = cfg.SAVE_MODEL_NAME;   

    private final KafkaConsumer<String, DataMessage> consumer;

    private final CoordinatorControl control;

    private final CustomLogger logger;

    private long t0;
    private long t1;
    private double lastActivitySeconds = 0.0;
    private long start_time = System.nanoTime();
    private long end = System.nanoTime();
    private long sumElapsedNs = 0;
    private int evaluation_count = 0;
    private float forwardPassNs = 0;
    private int countForwardPass = 0;

    private final Deque<DataMessage> carry = new ArrayDeque<>();

    private final String testStoreName;
    private KeyValueStore<String, ValueAndTimestamp<DataMessage>> testStore;
    private volatile List<DataMessage> cachedTestSet = null;

    private static final AtomicInteger INSTANCE_SEQ = new AtomicInteger(0);
    private final int instanceNo = INSTANCE_SEQ.incrementAndGet();
    private final String taskInstance = instanceNo + "@" + Integer.toHexString(System.identityHashCode(this));
    private String taskTag = "task=UNKNOWN";

    private static final int MIN_TEST_ROWS = 200;
    private static final long WAIT_SLEEP_MS = 100;
    private static final int WAIT_MAX_TRIES = 200; // 200 * 100ms = 20s max

    private int start;

    private int evalCursor = 0;                       // rolling start index into store
    private int cachedStoreSize = -1;                 // optional: track size changes

    // ================================================================================================================


    public CoordinatorProcessor(PsoModel globalModel, PsoModel bestGlobalModel, 
            long t0, long t1, String testStoreName, PsoModel preTrainedModel, int start) {

        this.logger = CustomLogger.getInstanceForCoordinator();

        this.t0 = t0;
        this.t1 = t1;

        this.control = CoordinatorControl.getInstance();

        this.globalModel = globalModel;
        this.bestGlobalModel = bestGlobalModel;

        if(cfg.USING_PRETRAINED_MODEL) {
            this.preTrainedModel = preTrainedModel;
            this.start = start;
        }

        this.globalPredictor = BatchPrediction.getInstanceForCoordinator(globalModel, bestGlobalModel, logger);

        this.testStoreName = testStoreName;

        Properties consumerProps = new Properties();
        consumerProps.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        consumerProps.put(ConsumerConfig.GROUP_ID_CONFIG, "pso-coordinator-eval-" + RUN_ID);
        consumerProps.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, DataMessageDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest"); // applies only when we dont commit the offset
        consumerProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");

        if (logger.isEnabled(2)) logger.log(taskInstance + " thread = " + Thread.currentThread().getName()
            + " TEST_TOPIC = " + TEST_TOPIC + " testStoreName = " + testStoreName);
            
        this.consumer = new KafkaConsumer<>(consumerProps);
        this.consumer.subscribe(Collections.singletonList(TEST_TOPIC));    
    }

    // ================================================================================================================

    @Override
    public void init(ProcessorContext<String, WeightsMessage> context) {    // this is output (Kout, Vout)
        this.context = context;
        this.testStore = (KeyValueStore<String, ValueAndTimestamp<DataMessage>>) context.getStateStore(testStoreName);
        this.taskTag = "task = " + context.taskId() + " thread = " + Thread.currentThread().getName();
        if (logger.isEnabled(2)) logger.log(taskInstance + " INIT " + taskTag 
                + " store = " + testStoreName);

    }

    // ================================================================================================================

    @Override
    public void process(Record<String, WeightsMessage> record) {

        start_time = System.nanoTime();
        updateTime();

        if (control.isStopRequested(-1)) {
            return;
        }

        if(evaluation_count == 0) {

            context.recordMetadata().ifPresent(meta -> {
                if (logger.isEnabled(2)) {
                    logger.log(taskInstance + ", Starting Meta Data: " + meta.topic() + 
                    ", Partition: " + meta.partition() + ", Offset: " + meta.offset());
                }
            });
        }

        WeightsMessage msg = record.value();
        if (msg == null) {
            return;
        }

        String workerId = String.valueOf(msg.workerId);
        
        float[] weights = msg.weights;
        if (weights == null) {
            return;
        }
        
        if(control.getBestTrainingAccuracy() < msg.accuracy) {
            control.setBestTrainingAccuracy(msg.accuracy);
        }

        weightsBuffer.put(workerId, weights);

        // Run only if all workers have reported their position 
        // if (logger.isEnabled(0)) logger.log("Which worker Id have already sent: " + weightsBuffer.keySet() + ", weightsBuffer.size() : " + weightsBuffer.size());

        if (weightsBuffer.size() == N_WORKERS) { // the particles of the workers should converge so asynchronous communication shouldnt matter
        
            float[] avgWeights = averageWeights(new ArrayList<>(weightsBuffer.values()));

            if(cfg.USING_PRETRAINED_MODEL) {
                Dl4jParamUtils.updateModelHead(globalModel, avgWeights, this.start);
            } else {
                Dl4jParamUtils.updateModel(globalModel, avgWeights);
            }

            // ======== evaluate accuracy of globalModel using BatchPrediction ========
            
            List<DataMessage> evalBatch;

            if (TEST_SIZE == -1) {
                List<DataMessage> full = loadAndCacheTestSet(MIN_TEST_ROWS);     // new using stateStore
                if (full == null || full.isEmpty()) {
                    if (logger.isEnabled(2)) logger.log(taskInstance + ", Cannot evaluate, Test set is null/empty.");
                    return;
                }

                int total = full.size();
                int batchSize = Math.min(cfg.TEST_BATCH_SIZE, total);
                if (evalCursor >= total) evalCursor = 0;
                int end = evalCursor + batchSize;

                if (end <= total) {
                    evalBatch = full.subList(evalCursor, end);
                } else {
                    List<DataMessage> tmp = new ArrayList<>(batchSize);
                    tmp.addAll(full.subList(evalCursor, total));
                    tmp.addAll(full.subList(0, end % total));
                    evalBatch = tmp;
                }

                evalCursor = (evalCursor + batchSize) % total;
            } else {

                evalBatch = readExactlyTestSizeBatch(TEST_SIZE);    // old, using Kafka consumer
                logConsumerOffsets();   
            }

            // if (TEST_SIZE == -1) {
            //     evalBatch = readNextBatchFromStore(cfg.TEST_BATCH_SIZE);
            // } else {
            //     // If you still want the Kafka-consumer path, keep it.
            //     // But for your state-store path, this is the rolling batch solution.
            //     evalBatch = readExactlyTestSizeBatch(Math.min(TEST_SIZE, cfg.TEST_BATCH_SIZE));
            // }
            // if (evalBatch == null || evalBatch.isEmpty()) {
            //     if (logger.isEnabled(2)) logger.log(taskInstance + 
            //             ", Cannot evaluate, Test set is null/empty.");
            //     return;
            // }

            float[] accLoss = globalPredictor.callPredictionsBatch(evalBatch, globalModel);  // inference / evaluate every time all workers current models arrive
                                                                                // monitor how training is going
            accuracy = accLoss[0];
            loss = accLoss[1];
            nSamples = (int) accLoss[2];
            nCorrect = (int) accLoss[3];
            forwardPassNs += accLoss[4];
            countForwardPass += 1;

            // update bestGlobalModelAccuracy + bestLoss ========================================================

            if(accuracy > control.getBestGlobalModelAccuracy()) {    

                if(cfg.USING_PRETRAINED_MODEL) {
                    Dl4jParamUtils.updateModelHead(bestGlobalModel, avgWeights, this.start);
                } else {
                    Dl4jParamUtils.updateModel(bestGlobalModel, avgWeights);
                }

                control.setBestGlobalModelAccuracy(accuracy);
                control.setBestGlobalModelLoss(loss);

                if (logger.isEnabled(1)) logger.log(taskInstance + 
                    ", New bestGlobalModel accuracy = " + control.getBestGlobalModelAccuracy());
            }

            if(loss < bestLoss) {    
                bestLoss = loss;
            }
            
            // ", process_count: " + process_count + " thread = " + Thread.currentThread().getName()
            if (logger.isEnabled(0)) logger.log(taskInstance + 
                        ") time: " + lastActivitySeconds + ", bestAccuracy: " + control.getBestGlobalModelAccuracy() + 
                        ", bestLoss: " + bestLoss + 
                        ", accuracy: " + accuracy + ", with nSamples: " + nSamples +
                        ", nCorrect: " + nCorrect + " loss: " + loss + 
                        ", weights sample: " + Dl4jParamUtils.sampleFlatSorted(avgWeights, SAMPLING_CONSTANT) +
                        ", bestTrainingAccuracy: " + control.getBestTrainingAccuracy());

            System.out.println(evaluation_count + 
                        ") time: " + lastActivitySeconds + ", bestAccuracy: " + control.getBestGlobalModelAccuracy() + 
                        ", bestLoss: " + bestLoss + 
                        ", accuracy: " + accuracy + ", with nSamples: " + nSamples +
                        ", nCorrect: " + nCorrect + " loss: " + loss + 
                        ", weights sample: " + Dl4jParamUtils.sampleFlatSorted(avgWeights, SAMPLING_CONSTANT) +
                        ", bestTrainingAccuracy: " + control.getBestTrainingAccuracy());

            if (control.getBestGlobalModelAccuracy() >= this.DESIRED_ACCURACY) {
                Dl4jParamUtils.saveModel(bestGlobalModel, SAVE_MODEL_NAME, start);
                control.requestStopFinal();
                return;
            }

            end = System.nanoTime();
            sumElapsedNs += (end - start_time);
            evaluation_count++;           

            //=================================================================================
            // weightsBuffer.clear();
            //=================================================================================
            // for(int workerId = weightsBuffer.keyes; i++) {
            //     if(control.isStopRequested(workerId) == false) {
            //         weightsBuffer[i].remove();
            //     } 
            // }
            //=================================================================================
            weightsBuffer.entrySet().removeIf(e -> {
                int wid;
                try {
                    wid = Integer.parseInt(e.getKey());
                } catch (NumberFormatException ex) {
                    return true;
                }
                return !control.isStopRequested(wid);   // if isStopRequested then dont remove it
            });
            //=================================================================================
        } 
    }

    //=========================================================================================================================

    private void updateTime() {
        
        t1 = System.nanoTime();
        lastActivitySeconds = Math.round(((t1 - t0) / 1_000_000_000.0) * 1000.0) / 1000.0;
    }

    //=========================================================================================================================

    private List<DataMessage> readExactlyTestSizeBatch(int testSize) {
        List<DataMessage> evalBatch = new ArrayList<>(testSize);

        // Use leftover test samples from previous poll
        while (evalBatch.size() < testSize && !carry.isEmpty()) {
            evalBatch.add(carry.removeFirst());
        }

        while (evalBatch.size() < testSize) {
            ConsumerRecords<String, DataMessage> records = consumer.poll(Duration.ofMillis(100));

            if (records.isEmpty()) {
                if (resetToBeginningIfAtEnd()) {
                    continue; 
                }
                continue;
            }

            for (ConsumerRecord<String, DataMessage> rec : records) {
                DataMessage dm = rec.value();
                if (dm == null) continue;

                if (evalBatch.size() < testSize) {
                    evalBatch.add(dm);
                } else {
                    carry.addLast(dm);
                }
            }
        }

        return evalBatch;
    }

    // ===============================================================================================

    private int approximateStoreSize() {
        int countLocal = 0;
        try (var it = testStore.all()) {
            while (it.hasNext()) {
                var kv = it.next();
                var vat = kv.value;
                if (vat != null && vat.value() != null) countLocal++;
            }
        }
        return countLocal;
    }

    //=========================================================================================================================

    private boolean resetToBeginningIfAtEnd() {

        Set<TopicPartition> asg = consumer.assignment();
        if (asg == null || asg.isEmpty()) {
            return false;
        }

        Map<TopicPartition, Long> ends = consumer.endOffsets(asg);

        boolean allAtEnd = true;
        for (TopicPartition tp : asg) {
            long pos = consumer.position(tp);
            long end = ends.getOrDefault(tp, -1L);

            // If end is unknown, treat as "not at end"
            if (end < 0) {
                allAtEnd = false;
                break;
            }

            if (pos < end) {
                allAtEnd = false;
                break;
            }
        }

        if (allAtEnd) {

            carry.clear();
            consumer.seekToBeginning(asg);
            consumer.poll(Duration.ZERO);
            if (logger.isEnabled(2)) logger.log(taskInstance + 
                ", Reached end-of-topic; resetting consumer to beginning (offset 0).");
            return true;
        }

        return false;
    }

    //=========================================================================================================================

    private List<DataMessage> loadAndCacheTestSet(int minRows) {

        if (cachedTestSet != null) return cachedTestSet;    // if already cached, just return the cache

        if (logger.isEnabled(2)) logger.log("Waiting on loadAndCacheTestSet");

        // wait until State Store has enough rows
        for (int tries = 0; tries < WAIT_MAX_TRIES; tries++) {
            int sz = approximateStoreSize();

            if (sz >= minRows) {
                if (logger.isEnabled(2)) logger.log("Breaking sleeping, estimated size is: " + 
                    sz + ", with minRows: " + minRows);
                break;
            };

            try {
                Thread.sleep(WAIT_SLEEP_MS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                if (logger.isEnabled(2)) logger.log(taskInstance + 
                        " interrupted while waiting for testStore");
                return null;
            }
        }

        // load all rows from store
        List<DataMessage> all = new ArrayList<>(1000);
        try (var it = testStore.all()) {    // the testStore
            while (it.hasNext()) {
                var kv = it.next();
                ValueAndTimestamp<DataMessage> vat = kv.value;
                if (vat != null && vat.value() != null) {
                    all.add(vat.value());
                }
            }
        }

        cachedTestSet = Collections.unmodifiableList(all);

        updateTime();

        if (logger.isEnabled(2)) logger.log(taskInstance + ", Timer: " + lastActivitySeconds + 
                ", cached TEST_STORE. Total rows = " + cachedTestSet.size());
        for (int i = 0; i < Math.min(5, cachedTestSet.size()); i++) {
            if (logger.isEnabled(2)) logger.log(taskInstance + 
                    ", TEST[" + i + "]: " + cachedTestSet.get(i));
        }

        if (logger.isEnabled(2)) logger.log("Done waiting on loadAndCacheTestSet, has been loaded into memory");
        System.out.println("[Coordinator] Test Samples have been loaded into memory, of length: " + cachedTestSet.size());
        
        if(false && cfg.USING_PRETRAINED_MODEL) {
            float[] accLoss;
            accLoss = globalPredictor.callPredictionsBatch(cachedTestSet, preTrainedModel);
            accuracy = accLoss[0];
            loss = accLoss[1];
            nSamples = (int) accLoss[2];
            nCorrect = (int) accLoss[3];

            if (logger.isEnabled(2)) logger.log("Report on preTrained Model accuracy: " + accuracy + ", with nSamples: " + nSamples +
                        ", nCorrect: " + nCorrect + " loss: " + loss);

            System.out.println("Report on preTrained Model: " + accuracy + ", with nSamples: " + nSamples +
                        ", nCorrect: " + nCorrect + " loss: " + loss);

            preTrainedModel.close();
            preTrainedModel.params().close();
            preTrainedModel = null;
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        }
        

        return cachedTestSet;
    }

    //=========================================================================================================================

    // private List<DataMessage> readNextBatchFromStore(int batchSize) {

    //     // Wait until store has at least batchSize rows (or at least something)
    //     for (int tries = 0; tries < WAIT_MAX_TRIES; tries++) {
    //         int sz = approximateStoreSize();
    //         if (sz >= Math.min(batchSize, MIN_TEST_ROWS)) {  // MIN_TEST_ROWS is your existing constant
    //             break;
    //         }
    //         try { Thread.sleep(WAIT_SLEEP_MS); }
    //         catch (InterruptedException e) { Thread.currentThread().interrupt(); return null; }
    //     }

    //     int storeSize = approximateStoreSize();
    //     if (storeSize <= 0) return null;

    //     // If store size changed a lot, keep cursor in range
    //     if (evalCursor >= storeSize) evalCursor = 0;

    //     // We will take [evalCursor, evalCursor + batchSize)
    //     // wrapping around at storeSize.
    //     int toTake = Math.min(batchSize, storeSize);

    //     // If you care about deterministic order, use sampleIndex ordering.
    //     // We'll do a 2-pass scan that collects the needed indices without loading everything.

    //     int startIdx = evalCursor;
    //     int endExclusive = evalCursor + toTake;

    //     List<DataMessage> out = new ArrayList<>(toTake);

    //     // Pass 1: collect from startIdx to storeSize-1
    //     int wantFrom1 = Math.min(toTake, storeSize - startIdx);
    //     if (wantFrom1 > 0) {
    //         collectBySortedIndexRange(out, startIdx, startIdx + wantFrom1);
    //     }

    //     // Pass 2: wrap around: collect from 0 to remaining-1
    //     int remaining = toTake - out.size();
    //     if (remaining > 0) {
    //         collectBySortedIndexRange(out, 0, remaining);
    //     }

    //     // Advance cursor for next time
    //     evalCursor = (evalCursor + toTake) % storeSize;

    //     return out;
    // }

    // //=========================================================================================================================

    // private void collectBySortedIndexRange(List<DataMessage> out, int startInclusive, int endExclusive) {

    //     // We need elements in order of sampleIndex.
    //     // We do this by collecting all keys/samplesIndex pairs, sorting, then reading only those in range.
    //     // If you have a better store key (like sampleIndex as key), we can make this O(batch) instead.

    //     List<DataMessage> tmp = new ArrayList<>();

    //     try (var it = testStore.all()) {
    //         while (it.hasNext()) {
    //             var kv = it.next();
    //             var vat = kv.value;
    //             if (vat == null || vat.value() == null) continue;
    //             tmp.add(vat.value());
    //         }
    //     }

    //     // Sort deterministically by sampleIndex (you already have it)
    //     tmp.sort(Comparator.comparingInt(dm -> dm.sampleIndex));

    //     int n = tmp.size();
    //     int s = Math.max(0, Math.min(startInclusive, n));
    //     int e = Math.max(0, Math.min(endExclusive, n));

    //     for (int i = s; i < e; i++) {
    //         out.add(tmp.get(i));
    //     }
    // }

    //=========================================================================================================================

    private void logConsumerOffsets() {
        try {
            Set<TopicPartition> asg = consumer.assignment();

            StringBuilder sb = new StringBuilder();
            sb.append("Consumer position: ");

            Map<TopicPartition, Long> ends = consumer.endOffsets(asg);

            for (TopicPartition tp : asg) {
                long pos = consumer.position(tp);   // current offset (position == offset)
                long end = ends.getOrDefault(tp, -1L);

                sb.append("[")
                    .append(tp.topic()).append("-").append(tp.partition())
                    .append(" pos = ").append(pos)
                    .append(" end = ").append(end)
                    .append("] ");
            }

            if (logger.isEnabled(1)) logger.log(sb.toString());
        } catch (Exception e) {
            if (logger.isEnabled(2)) logger.log(taskInstance + 
                ", Coordinator failed to log consumer offsets: " + e.getMessage());
        }
    }
   
    //=========================================================================================================================

    private static float[] averageWeights(List<float[]> bufs) {
        if (bufs == null || bufs.isEmpty()) return new float[0];

        int numWorkers = bufs.size();
        int len = bufs.get(0).length;
        float[] average = new float[len];

        for (float[] arr : bufs) {
            for (int i = 0; i < len; i++) {
                average[i] += arr[i];
            }
        }
        
        for (int i = 0; i < len; i++) {
            average[i] /= numWorkers;
        }
        return average;
    }
    
    //========================================================================================================================

    public static class DesiredAccuracyReachedException extends RuntimeException {
        public DesiredAccuracyReachedException(String message) {
            super(message);
        }
    }
    
    // ==================================================================================================================================

    @Override
    public void close() {
        try {
            consumer.wakeup();                // breaks poll safely
        } catch (Exception ignored) {}

        try {
            consumer.close(Duration.ofSeconds(5));
        } catch (Exception ignored) {

        }

        Dl4jParamUtils.saveModel(bestGlobalModel, SAVE_MODEL_NAME, this.start);         // save final solution
        double avgMs = (sumElapsedNs / 1_000_000.0) / evaluation_count;
        double avgForwardPassMs = forwardPassNs / countForwardPass;

        if (logger.isEnabled(2)) logger.log(taskInstance + 
                ", average elapsed time per batch: " + String.format("%.3f ms", avgMs)
                + " over " + evaluation_count + " batches" + ", average forwardPassMs: " 
                + avgForwardPassMs);

        this.logger.flush();
    }
}

    // ===============================================================================================

    // private List<DataMessage> getAllTestRowsFromStoreOnce() {

    //     if (cachedTestSet != null) return cachedTestSet;

    //     // Wait for the global store to populate (size stabilizes)
    //     int lastSize = -1;
    //     int stableCount = 0;

    //     for (int tries = 0; tries < 50; tries++) { // ~50 * 100ms = 5s max
    //         int sz = approximateStoreSize();
    //         if (sz == lastSize && sz > 0) {
    //             stableCount++;
    //             if (stableCount >= 5) break; // stable for 5 checks
    //         } else {
    //             stableCount = 0;
    //             lastSize = sz;
    //         }

    //         try { Thread.sleep(50);  } catch (InterruptedException e) { Thread.currentThread().interrupt(); break; }
    //     }

    //     List<DataMessage> all = new ArrayList<>();
    //     try (var it = testStore.all()) {
    //         while (it.hasNext()) {
    //             var kv = it.next();
    //             ValueAndTimestamp<DataMessage> vat = kv.value;
    //             if (vat != null && vat.value() != null) {
    //                 all.add(vat.value());
    //             }
    //         }
    //     }

    //     all.sort(Comparator.comparingInt(dm -> dm.sampleIndex));
    //     cachedTestSet = Collections.unmodifiableList(all);

    //     if (logger.isEnabled(0)) logger.log(taskInstance + ", Timer: " + lastActivitySeconds + ", loaded TEST_STORE into memory. Total test rows = " + cachedTestSet.size());
    //     for (int i = 0; i < Math.min(5, cachedTestSet.size()); i++) {
    //         if (logger.isEnabled(0)) logger.log(taskInstance + ", TEST[" + i + "]: " + cachedTestSet.get(i));
    //     }

    //     return cachedTestSet;
    // }

    // private int approximateStoreSize() {
    //     int count = 0;
    //     try (var it = testStore.all()) {
    //         while (it.hasNext()) {
    //             var kv = it.next();
    //             var vat = kv.value;
    //             if (vat != null && vat.value() != null) count++;
    //         }
    //     }
    //     return count;
    // }

    // ======================================================================================================================

    // private List<DataMessage> loadAllTestDataOnce() {

    //     if (cachedTestSetLoaded && cachedTestSet != null) {
    //         if (logger.isEnabled(0)) logger.log(taskInstance + ", Using cached Test Set");
    //         return cachedTestSet;
    //     }

    //     consumer.poll(Duration.ZERO);
    //     Set<TopicPartition> asg = consumer.assignment();

    //     if (asg == null || asg.isEmpty()) {
    //         // poll again to get assignment
    //         consumer.poll(Duration.ofMillis(100));
    //         asg = consumer.assignment();
    //     }
    //     if (asg == null || asg.isEmpty()) {
    //         if (logger.isEnabled(0)) logger.log(taskInstance + ", Could not get assignment for TEST_TOPIC; cannot cache test set.");
    //         return null;
    //     }

    //     consumer.seekToBeginning(asg);
    //     consumer.poll(Duration.ZERO);

    //     Map<TopicPartition, Long> ends = consumer.endOffsets(asg);

    //     List<DataMessage> all = new ArrayList<>(4096);

    //     while (true) {   // Read until all partitions reach end offsets

    //         ConsumerRecords<String, DataMessage> records = consumer.poll(Duration.ofMillis(200));

    //         for (ConsumerRecord<String, DataMessage> rec : records) {
    //             DataMessage dm = rec.value();
    //             if (dm != null) all.add(dm);
    //         }

    //         boolean allAtEnd = true;
    //         for (TopicPartition tp : asg) {
    //             long pos = consumer.position(tp);
    //             long end = ends.getOrDefault(tp, -1L);

    //             if (pos < end) {    // if not yet at end
    //                 allAtEnd = false;
    //                 break;
    //             }
    //         }

    //         if (allAtEnd) break;    // if at end break
    //     }

    //     cachedTestSet = Collections.unmodifiableList(all);
    //     cachedTestSetLoaded = true;

    //     if (logger.isEnabled(0)) logger.log(taskInstance + ", Cached full TEST_TOPIC into memory. Total test rows = " + cachedTestSet.size());
    //     if (logger.isEnabled(0)) logger.log(taskInstance + ", First 5 TEST samples:");
    //     for (int i = 0; i < Math.min(5, cachedTestSet.size()); i++) {
    //         if (logger.isEnabled(0)) logger.log(taskInstance + ", TEST[" + i + "]: " + cachedTestSet.get(i).toString());
    //     }

    //     return cachedTestSet;
    // }

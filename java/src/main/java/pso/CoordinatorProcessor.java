package pso;

import org.apache.kafka.streams.processor.PunctuationType;
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
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

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

    // private KafkaConsumer<String, DataMessage> consumer;

    private final CoordinatorControl control;

    private final CustomLogger logger;

    private long t0;
    private AtomicLong t_actually_started = new AtomicLong(t0);
    private long t1;
    private double lastActivitySeconds = 0.0;
    private long start_time;
    private long end = System.nanoTime();
    private long sumElapsedNs = 0;
    private int evaluation_count = 0;
    private float forwardPassNs = 0;

    private static final long IDLE_MS = cfg.IDLE_MS; 
    private static final long CHECK_EVERY_MS = 100; 
    private static final long IDLE_GRACE_MS = 8000;

    private final Deque<DataMessage> carry = new ArrayDeque<>();

    private final String testStoreName;
    private KeyValueStore<String, ValueAndTimestamp<DataMessage>> testStore;
    private volatile List<DataMessage> cachedTestSet = null;

    private static final AtomicInteger INSTANCE_SEQ = new AtomicInteger(0);
    private final int instanceNo = INSTANCE_SEQ.incrementAndGet();
    private final String taskInstance = instanceNo + "@" + Integer.toHexString(System.identityHashCode(this));
    private String taskTag = "task=UNKNOWN";

    private static final int MIN_TEST_ROWS = 200;
    private static final long WAIT_SLEEP_MS = 50;
    private static final int WAIT_MAX_TRIES = 50; // 200 * 100ms = 20s max

    private int start;

    private int evalCursor = 0;                       // rolling start index into store
    private int cachedStoreSize = -1;                 // optional: track size changes

    private int roundsWithoutImprovement = 0;

    private long start_waiting_for_test = System.nanoTime();
    private long end_waiting_for_test = System.nanoTime();

    // ================================================================================================================

    public CoordinatorProcessor(PsoModel globalModel, PsoModel bestGlobalModel, 
            long t0, AtomicLong t_actually_started, long t1, String testStoreName, PsoModel preTrainedModel, int start) {

        this.logger = CustomLogger.getInstanceForCoordinator();

        this.t0 = t0;
        this.t_actually_started = t_actually_started;
        this.t1 = t1;
        updateTime();
        logger.log("Starting Delay 0: " + (System.nanoTime() - this.t0) / 1_000_000_000.0);

        this.control = CoordinatorControl.getInstance();

        this.globalModel = globalModel;
        this.bestGlobalModel = bestGlobalModel;

        if(cfg.USING_PRETRAINED_MODEL) {
            this.preTrainedModel = preTrainedModel;
            this.start = start;
        }

        this.globalPredictor = BatchPrediction.getInstanceForCoordinator(globalModel, bestGlobalModel, logger);

        this.testStoreName = testStoreName;
        if (logger.isEnabled(2)) logger.log(taskInstance + " thread = " + Thread.currentThread().getName()
            + " TEST_TOPIC = " + TEST_TOPIC + " testStoreName = " + testStoreName);

        // Properties consumerProps = new Properties();
        // consumerProps.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, cfg.KAFKA_HOST);
        // consumerProps.put(ConsumerConfig.GROUP_ID_CONFIG, "pso-coordinator-eval-" + RUN_ID);
        // consumerProps.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        // consumerProps.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, DataMessageDeserializer.class.getName());
        // consumerProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest"); // applies only when we dont commit the offset
        // consumerProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");            
        // this.consumer = new KafkaConsumer<>(consumerProps);
        // this.consumer.subscribe(Collections.singletonList(TEST_TOPIC));    

        this.start_time = System.nanoTime();
    }

    // ================================================================================================================

    @Override
    public void init(ProcessorContext<String, WeightsMessage> context) {    // this is output (Kout, Vout)
        this.context = context;
        this.testStore = (KeyValueStore<String, ValueAndTimestamp<DataMessage>>) context.getStateStore(testStoreName);
        this.taskTag = "task = " + context.taskId() + " thread = " + Thread.currentThread().getName();
        if (logger.isEnabled(2)) logger.log(taskInstance + " INIT " + taskTag 
                + " store = " + testStoreName);
        
        context.schedule(Duration.ofMillis(CHECK_EVERY_MS), PunctuationType.WALL_CLOCK_TIME, timestamp -> {

            long sinceStartNs = System.nanoTime() - t0;
            if (sinceStartNs < TimeUnit.MILLISECONDS.toNanos(IDLE_GRACE_MS)) {
                return; 
            }

            long idleNs = System.nanoTime() - t1;
            
            if (idleNs >= TimeUnit.MILLISECONDS.toNanos(IDLE_MS)) {
                if (logger.isEnabled(2)) logger.log(taskInstance + 
                    ", Closed, because of idleness for " + (idleNs / 1_000_000) + " ms");
                System.out.println("[Coordinator] Closed, because of idleness for " + (idleNs / 1_000_000) + " ms");
                CoordinatorControl.getInstance().requestStopFinal();
            }
        });
    }

    // ================================================================================================================

    @Override
    public void process(Record<String, WeightsMessage> record) {
        if (control.isStopRequested(-1)) {
            // onAllWorkersReported();
            return;
        }

        if(evaluation_count == 0) {
            t_actually_started.set(System.nanoTime());
            context.recordMetadata().ifPresent(meta -> {
                if (logger.isEnabled(2)) {
                    logger.log(taskInstance + ", Starting Meta Data: " + meta.topic() + 
                    ", Partition: " + meta.partition() + ", Offset: " + meta.offset());
                    logger.log("Stating Delay 1: " + (System.nanoTime() - this.t0) / 1_000_000_000.0);
                    logger.log("Starting Delay 2: " + (System.nanoTime() - this.start_time) / 1_000_000_000.0);
                }
            });
        }

        start_time = System.nanoTime();
        updateTime();
        control.processedAtLeastOne = true;

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
        
        if (weightsBuffer.size() == N_WORKERS) {
            onAllWorkersReported();
        }
    }

//=========================================================================================================================
//=========================================================================================================================
//=========================================================================================================================

public void onAllWorkersReported() {

    float[] avgWeights = averageWeights(new ArrayList<>(weightsBuffer.values()));

    if(cfg.USING_PRETRAINED_MODEL) {
        Dl4jParamUtils.updateModelHead(globalModel, avgWeights, this.start);
    } else {
        Dl4jParamUtils.updateModel(globalModel, avgWeights);
    }            

    List<DataMessage> evalBatch = null;

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
    } 
    
    // else {

    //     evalBatch = readExactlyTestSizeBatch(TEST_SIZE);    // old, using Kafka consumer
    //     logConsumerOffsets();   
    // }
    
    float[] accLoss = globalPredictor.callPredictionsBatch(evalBatch, globalModel, false);  // inference / evaluate every time all workers current models arrive
                                                                        // monitor how training is going
    accuracy = accLoss[0];
    loss = accLoss[1];
    nSamples = (int) accLoss[2];
    nCorrect = (int) accLoss[3];
    forwardPassNs += accLoss[4];

    control.accuracyValues.add(new AccuracyPoint(lastActivitySeconds, accuracy));
    
    // update bestGlobalModelAccuracy + bestLoss ========================================================

    if(accuracy > control.getBestGlobalModelAccuracy()) {    

        if(cfg.USING_PRETRAINED_MODEL) {
            Dl4jParamUtils.updateModelHead(bestGlobalModel, avgWeights, this.start);
        } else {
            Dl4jParamUtils.updateModel(bestGlobalModel, avgWeights);
        }

        control.setBestGlobalModelAccuracy(accuracy);
        control.setBestGlobalModelLoss(loss);
        roundsWithoutImprovement = 0;

        if (logger.isEnabled(1)) logger.log(taskInstance + 
            ", New bestGlobalModel accuracy = " + control.getBestGlobalModelAccuracy());
    } else {
        roundsWithoutImprovement++;
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
                ", bestTrainingAccuracy: " + control.getBestTrainingAccuracy() +
                ", noImprovementRounds: " + roundsWithoutImprovement);

    // System.out.println(evaluation_count + 
    //             ") time: " + lastActivitySeconds + ", bestAccuracy: " + control.getBestGlobalModelAccuracy() + 
    //             ", bestLoss: " + bestLoss + 
    //             ", accuracy: " + accuracy + ", with nSamples: " + nSamples +
    //             ", nCorrect: " + nCorrect + " loss: " + loss + 
    //             ", weights sample: " + Dl4jParamUtils.sampleFlatSorted(avgWeights, SAMPLING_CONSTANT) +
    //             ", bestTrainingAccuracy: " + control.getBestTrainingAccuracy() +
    //             ", noImprovementRounds: " + roundsWithoutImprovement);

    // only accuracies
    System.out.println(evaluation_count + 
                ") time: " + lastActivitySeconds + ", bestAccuracy: " + control.getBestGlobalModelAccuracy() + 
                ", accuracy: " + accuracy + ", loss: " + loss + 
                ", bestTrainingAccuracy: " + control.getBestTrainingAccuracy() +
                ", noImprovementRounds: " + roundsWithoutImprovement);

    if (control.getBestGlobalModelAccuracy() >= this.DESIRED_ACCURACY) {
        System.out.println("Stopped, because DESIRED_ACCURACY reached");
        control.requestStopFinal();
        return;
    }

    if(cfg.EARLY_STOPPING) {
        if (roundsWithoutImprovement >= cfg.MAX_NO_IMPROVEMENT_ROUNDS) {
            if (logger.isEnabled(1)) logger.log(taskInstance +
                ", Early stopping: bestAccuracy has not improved for " +
                roundsWithoutImprovement + " rounds");
            System.out.println("[Coordinator] Early stopping: bestAccuracy has not improved for " +
                roundsWithoutImprovement + " rounds");
            control.requestStopFinal();
            return;
        }
    }

    end = System.nanoTime();
    sumElapsedNs += (end - start_time);
    evaluation_count++;           

    weightsBuffer.entrySet().removeIf(e -> {    // if isStopRequested then dont remove it
        int wid;
        try {
            wid = Integer.parseInt(e.getKey());
        } catch (NumberFormatException ex) {
            return true;
        }

        return !control.isStopRequested(wid);   // if isStopRequested then dont remove it
    });
} 

//=========================================================================================================================
//=========================================================================================================================
//=========================================================================================================================

    private void updateTime() {
        t1 = System.nanoTime();
        lastActivitySeconds = Math.round(((t1 - t0) / 1_000_000_000.0) * 1000.0) / 1000.0;
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

    private List<DataMessage> loadAndCacheTestSet(int minRows) {

        if (cachedTestSet != null) return cachedTestSet;    // if already cached, just return the cache
        start_waiting_for_test = System.nanoTime();

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
        end_waiting_for_test = System.nanoTime();
        updateTime();

        if (logger.isEnabled(2)) logger.log(taskInstance + ", Timer: " + lastActivitySeconds + 
                ", cached TEST_STORE. Total rows = " + cachedTestSet.size());

        for (int i = 0; i < Math.min(5, cachedTestSet.size()); i++) {
            if (logger.isEnabled(2)) logger.log(taskInstance + 
                    ", TEST[" + i + "]: " + cachedTestSet.get(i));
        }

        if (logger.isEnabled(2)) logger.log("Done waiting on loadAndCacheTestSet, has been loaded into memory");
        System.out.println("[Coordinator] Test Samples have been loaded into memory, of length: " + cachedTestSet.size());
        

        if(cfg.EVALUATE_PRETRAINED && cfg.USING_PRETRAINED_MODEL && cfg.TESTABLE_PRETRAINED_MODEL) {
            float[] accLoss;
            accLoss = globalPredictor.callPredictionsBatch(cachedTestSet, preTrainedModel, true);
            accuracy = accLoss[0];
            loss = accLoss[1];
            nSamples = (int) accLoss[2];
            nCorrect = (int) accLoss[3];
            
            control.setPretrainedAccuracy(accuracy);
            
            if (logger.isEnabled(2)) logger.log("Report on preTrained Model accuracy: " + accuracy + ", with nSamples: " + nSamples +
                        ", nCorrect: " + nCorrect + " loss: " + loss);

            System.out.println("Report on preTrained Model: " + accuracy + ", with nSamples: " + nSamples +
                        ", nCorrect: " + nCorrect + " loss: " + loss);

            preTrainedModel.close();
            preTrainedModel.params().close();
            preTrainedModel = null;
        }
        
        return cachedTestSet;
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

        if(!cfg.SAVE_MODEL_NAME.equals("no-save")) {
            Dl4jParamUtils.saveModel(bestGlobalModel, SAVE_MODEL_NAME, this.start); // save final solution
        }

        double avgMs = (sumElapsedNs / 1_000_000.0) / evaluation_count;
        double avgForwardPassMs = forwardPassNs / evaluation_count;

        if (logger.isEnabled(2)) logger.log(taskInstance + 
                ", average elapsed time per batch: " + String.format("%.3f ms", avgMs)
                + " over " + evaluation_count + " batches" + ", average forwardPassMs: " 
                + avgForwardPassMs);

        if(control.getBestGlobalModelAccuracy() == -1f) { // coordinator never evaluated local state
            if(weightsBuffer.size() != 0) {
                onAllWorkersReported();
            } else {
                if(logger.isEnabled(2)) logger.log("weightsBuffer empty, skipping final evaluation");
                System.out.println("[Coordinator] weightsBuffer empty, skipping final evaluation");
            }
        }   
        double initialTestLoadMs = (end_waiting_for_test - start_waiting_for_test) / 1_000_000.0;
        double initialTestLoadSec = (end_waiting_for_test - start_waiting_for_test) / 1_000_000_000.0;

        if (end_waiting_for_test > start_waiting_for_test) {
            if (logger.isEnabled(2)) {
                logger.log(taskInstance + ", initial test set wait/load delay: "
                        + String.format("%.3f ms", initialTestLoadMs)
                        + " (" + String.format("%.3f s", initialTestLoadSec) + ")");
            }

            System.out.println("[Coordinator] Initial test set wait/load delay: "
                    + String.format("%.3f ms", initialTestLoadMs)
                    + " (" + String.format("%.3f s", initialTestLoadSec) + ")");
        }
        this.logger.flush();
    }
    //=========================================================================================================================

    // private List<DataMessage> readExactlyTestSizeBatch(int testSize) {
    //     List<DataMessage> evalBatch = new ArrayList<>(testSize);

    //     // Use leftover test samples from previous poll
    //     while (evalBatch.size() < testSize && !carry.isEmpty()) {
    //         evalBatch.add(carry.removeFirst());
    //     }

    //     while (evalBatch.size() < testSize) {
    //         ConsumerRecords<String, DataMessage> records = consumer.poll(Duration.ofMillis(100));

    //         if (records.isEmpty()) {
    //             if (resetToBeginningIfAtEnd()) {
    //                 continue; 
    //             }
    //             continue;
    //         }

    //         for (ConsumerRecord<String, DataMessage> rec : records) {
    //             DataMessage dm = rec.value();
    //             if (dm == null) continue;

    //             if (evalBatch.size() < testSize) {
    //                 evalBatch.add(dm);
    //             } else {
    //                 carry.addLast(dm);
    //             }
    //         }
    //     }

    //     return evalBatch;
    // }

    //=========================================================================================================================

    // private boolean resetToBeginningIfAtEnd() {

    //     Set<TopicPartition> asg = consumer.assignment();
    //     if (asg == null || asg.isEmpty()) {
    //         return false;
    //     }

    //     Map<TopicPartition, Long> ends = consumer.endOffsets(asg);

    //     boolean allAtEnd = true;
    //     for (TopicPartition tp : asg) {
    //         long pos = consumer.position(tp);
    //         long end = ends.getOrDefault(tp, -1L);

    //         // If end is unknown, treat as "not at end"
    //         if (end < 0) {
    //             allAtEnd = false;
    //             break;
    //         }

    //         if (pos < end) {
    //             allAtEnd = false;
    //             break;
    //         }
    //     }

    //     if (allAtEnd) {

    //         carry.clear();
    //         consumer.seekToBeginning(asg);
    //         consumer.poll(Duration.ZERO);
    //         if (logger.isEnabled(2)) logger.log(taskInstance + 
    //             ", Reached end-of-topic; resetting consumer to beginning (offset 0).");
    //         return true;
    //     }

    //     return false;
    // }

    //=========================================================================================================================

    // private List<DataMessage> readNextBatchFromStore(int batchSize) {

    //     for (int tries = 0; tries < WAIT_MAX_TRIES; tries++) {
    //         int sz = approximateStoreSize();
    //         if (sz >= Math.min(batchSize, MIN_TEST_ROWS)) {  
    //             break;
    //         }
    //         try { Thread.sleep(WAIT_SLEEP_MS); }
    //         catch (InterruptedException e) { Thread.currentThread().interrupt(); return null; }
    //     }

    //     int storeSize = approximateStoreSize();
    //     if (storeSize <= 0) return null;

    //     if (evalCursor >= storeSize) evalCursor = 0;
    //     int toTake = Math.min(batchSize, storeSize);

    //     int startIdx = evalCursor;
    //     int endExclusive = evalCursor + toTake;

    //     List<DataMessage> out = new ArrayList<>(toTake);

    //     int wantFrom1 = Math.min(toTake, storeSize - startIdx);
    //     if (wantFrom1 > 0) {
    //         collectBySortedIndexRange(out, startIdx, startIdx + wantFrom1);
    //     }

    //     int remaining = toTake - out.size();
    //     if (remaining > 0) {
    //         collectBySortedIndexRange(out, 0, remaining);
    //     }

    //     evalCursor = (evalCursor + toTake) % storeSize;

    //     return out;
    // }

    // private void collectBySortedIndexRange(List<DataMessage> out, int startInclusive, int endExclusive) {

    //     List<DataMessage> tmp = new ArrayList<>();

    //     try (var it = testStore.all()) {
    //         while (it.hasNext()) {
    //             var kv = it.next();
    //             var vat = kv.value;
    //             if (vat == null || vat.value() == null) continue;
    //             tmp.add(vat.value());
    //         }
    //     }

    //     tmp.sort(Comparator.comparingInt(dm -> dm.sampleIndex));

    //     int n = tmp.size();
    //     int s = Math.max(0, Math.min(startInclusive, n));
    //     int e = Math.max(0, Math.min(endExclusive, n));

    //     for (int i = s; i < e; i++) {
    //         out.add(tmp.get(i));
    //     }
    // }

    //=========================================================================================================================

    // private void logConsumerOffsets() {
    //     try {
    //         Set<TopicPartition> asg = consumer.assignment();

    //         StringBuilder sb = new StringBuilder();
    //         sb.append("Consumer position: ");

    //         Map<TopicPartition, Long> ends = consumer.endOffsets(asg);

    //         for (TopicPartition tp : asg) {
    //             long pos = consumer.position(tp);   // current offset (position == offset)
    //             long end = ends.getOrDefault(tp, -1L);

    //             sb.append("[")
    //                 .append(tp.topic()).append("-").append(tp.partition())
    //                 .append(" pos = ").append(pos)
    //                 .append(" end = ").append(end)
    //                 .append("] ");
    //         }

    //         if (logger.isEnabled(1)) logger.log(sb.toString());
    //     } catch (Exception e) {
    //         if (logger.isEnabled(2)) logger.log(taskInstance + 
    //             ", Coordinator failed to log consumer offsets: " + e.getMessage());
    //     }
    // }
   
}

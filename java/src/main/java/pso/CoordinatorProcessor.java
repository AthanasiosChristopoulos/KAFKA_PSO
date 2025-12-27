package pso;

import org.apache.kafka.streams.processor.api.ContextualProcessor;
import org.apache.kafka.streams.processor.api.Record;
import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.streams.processor.api.ProcessorContext;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
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
    
    private float gBestAccuracy = 0f;
    private float gBestLoss = 10000f;

    private final MultiLayerNetwork globalModel; // x_g , current model
    private final MultiLayerNetwork bestGlobalModel; 
    private float accuracy = -1f;
    private float loss = 10000f;
    private int nSamples = 0;
    private int nCorrect = 0;
    private float bestGlobalModelAccuracy = -1f;
    private float bestLoss = 10000f;

    private final BatchPrediction globalPredictor;

    private static Config cfg = Config.getInstance();
    private final int N_WORKERS = cfg.N_WORKERS;
    private final String TEST_TOPIC = cfg.TEST_TOPIC;
    private final int TEST_SIZE = cfg.TEST_SIZE;
    private final float DESIRED_ACCURACY = cfg.DESIRED_ACCURACY;
    private final String RUN_ID = cfg.RUN_ID;   
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT; 

    private final KafkaConsumer<String, DataMessage> consumer;

    private int count = 0;

    private final CoordinatorControl control;

    private final CustomLogger logger;

    private long t0;
    private long t1;
    private double lastActivitySeconds = 0.0;

    private final Deque<DataMessage> carry = new ArrayDeque<>();

    private int test_count = 0;

    private final String testStoreName;
    private KeyValueStore<String, ValueAndTimestamp<DataMessage>> testStore;
    private volatile List<DataMessage> cachedTestSet = null;

    private static final AtomicInteger INSTANCE_SEQ = new AtomicInteger(0);
    private final int instanceNo = INSTANCE_SEQ.incrementAndGet();
    private final String taskInstance = "CoordinatorProcessor#" + instanceNo + "@" + Integer.toHexString(System.identityHashCode(this));
    private String taskTag = "task=UNKNOWN";


    // ================================================================================================================


    public CoordinatorProcessor(MultiLayerNetwork globalModel, MultiLayerNetwork bestGlobalModel, long t0, long t1, String testStoreName) {
        
        this.t0 = t0;
        this.t1 = t1;

        this.control = CoordinatorControl.getInstance();

        this.logger = CustomLogger.getInstanceForCoordinator();

        this.globalModel = globalModel;
        this.bestGlobalModel = bestGlobalModel;

        this.globalPredictor = BatchPrediction.getInstanceForCoordinator(globalModel, bestGlobalModel, logger);

        this.testStoreName = testStoreName;

        Properties consumerProps = new Properties();
        consumerProps.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        consumerProps.put(ConsumerConfig.GROUP_ID_CONFIG, "pso-coordinator-eval-" + RUN_ID);
        consumerProps.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, DataMessageDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest"); // applies only when we dont commit the offset
        consumerProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");
        // logger.log("TEST_TOPIC: " + TEST_TOPIC);

        logger.log(taskInstance + " thread=" + Thread.currentThread().getName()
            + " TEST_TOPIC=" + TEST_TOPIC + " testStoreName=" + testStoreName);
            
        this.consumer = new KafkaConsumer<>(consumerProps);
        this.consumer.subscribe(Collections.singletonList(TEST_TOPIC));    
    }

    @Override
    public void init(ProcessorContext<String, WeightsMessage> context) {    // this is output (Kout, Vout)
        this.context = context;
        this.testStore = (KeyValueStore<String, ValueAndTimestamp<DataMessage>>) context.getStateStore(testStoreName);
        this.taskTag = "task=" + context.taskId() + " thread=" + Thread.currentThread().getName();
        logger.log(taskInstance + " INIT " + taskTag + " store=" + testStoreName);


    }

    // ================================================================================================================

    @Override
    public void process(Record<String, WeightsMessage> record) {

        if (control.isStopRequested(-1)) return;
        
        if(count == 0) {
            context.recordMetadata().ifPresent(meta -> 
                logger.log("Starting Meta Data: " + meta.topic() + ", Partition: " + meta.partition() + ", Offset: " + meta.offset())
            );
        }

        count = count + 1;

        WeightsMessage msg = record.value();
        if (msg == null) {
            return;
        }

        String workerId = String.valueOf(msg.idWorker);
        
        updateTime();

        // logger.log("Time: " + lastActivitySeconds + " current Position message with msgIndex " + msg.msgIndex + ", from worker " + workerId);

        float[] weights = msg.weights;
        if (weights == null) {
            return;
        }
    
        weightsBuffer.put(workerId, weights);

        // Run only if all workers have reported their position 
        if (weightsBuffer.size() == N_WORKERS) { // the particles of the workers should converge so asynchronous communication shouldnt matter
        
            float[] avgWeights = averageWeights(new ArrayList<>(weightsBuffer.values()));

            Dl4jParamUtils.updateModel(globalModel, avgWeights);

            // ======== evaluate accuracy of globalModel using BatchPrediction ========
            
            List<DataMessage> evalBatch;

            if (TEST_SIZE == -1) {
                evalBatch = getAllTestRowsFromStoreOnce();
                if (evalBatch == null || evalBatch.isEmpty()) {
                    logger.log("TEST_SIZE=-1 but cached test set is null/empty. Cannot evaluate.");
                    return;
                }
            } else {
                evalBatch = readExactlyTestSizeBatch(TEST_SIZE);
                if (evalBatch == null) {
                    System.out.println("Test Records run out. Something is wrong");
                    return;
                }
                logConsumerOffsets();   
            }

            float[] accLoss = globalPredictor.callPredictionsBatch(evalBatch);
            accuracy = accLoss[0];
            loss = accLoss[1];
            nSamples = (int) accLoss[2];
            nCorrect = (int) accLoss[3];

            // update bestGlobalModelAccuracy + bestLoss ========================================================

            if(accuracy > bestGlobalModelAccuracy) {    
                Dl4jParamUtils.updateModel(bestGlobalModel, avgWeights);
                bestGlobalModelAccuracy = accuracy;
                logger.log("New bestGlobalModel accuracy = " + bestGlobalModelAccuracy);
            }

            if(loss < bestLoss) {    
                bestLoss = loss;
            }
            
            logger.log(taskInstance + " thread=" + Thread.currentThread().getName()
            + test_count + ") time: " + lastActivitySeconds + 
                        ", bestAccuracy: " + bestGlobalModelAccuracy + ", bestLoss: " + bestLoss + 
                        ", accuracy: " + accuracy + ", with nSamples: " + nSamples
                        + ", nCorrect: " + nCorrect + " and loss: " + loss + 
                        ", and weights sample: " + Dl4jParamUtils.sampleFlatSorted(avgWeights, SAMPLING_CONSTANT));

            System.out.println(test_count + ") time: " + lastActivitySeconds + 
                        ", bestAccuracy: " + bestGlobalModelAccuracy + ", bestLoss: " + bestLoss + 
                        ", accuracy: " + accuracy + " and loss: " + loss + 
                        ", and weights sample: " + Dl4jParamUtils.sampleFlatSorted(avgWeights, SAMPLING_CONSTANT));

            test_count++;

            if (bestGlobalModelAccuracy >= this.DESIRED_ACCURACY) {
                Dl4jParamUtils.saveModel(bestGlobalModel);
                control.requestStopFinal();
                return;
            }

            weightsBuffer.clear();
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
    }

    //=========================================================================================================================

    private void updateTime() {
        t1 = System.nanoTime();
        lastActivitySeconds = Math.round(((t1 - t0) / 1_000_000_000.0) * 10.0) / 10.0;
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

    //=========================================================================================================================

    private List<DataMessage> getAllTestRowsFromStoreOnce() {

        if (cachedTestSet != null) return cachedTestSet;

        // Wait for the global store to populate (size stabilizes)
        int lastSize = -1;
        int stableCount = 0;

        for (int tries = 0; tries < 50; tries++) { // ~50 * 100ms = 5s max
            int sz = approximateStoreSize();
            if (sz == lastSize && sz > 0) {
                stableCount++;
                if (stableCount >= 5) break; // stable for 5 checks
            } else {
                stableCount = 0;
                lastSize = sz;
            }

            try { Thread.sleep(100); } catch (InterruptedException e) { Thread.currentThread().interrupt(); break; }
        }

        List<DataMessage> all = new ArrayList<>();
        try (var it = testStore.all()) {
            while (it.hasNext()) {
                var kv = it.next();
                ValueAndTimestamp<DataMessage> vat = kv.value;
                if (vat != null && vat.value() != null) {
                    all.add(vat.value());
                }
            }
        }

        all.sort(Comparator.comparingInt(dm -> dm.sampleIndex));
        cachedTestSet = Collections.unmodifiableList(all);

        logger.log("Timer: " + lastActivitySeconds + ", loaded TEST_STORE into memory. Total test rows = " + cachedTestSet.size());
        for (int i = 0; i < Math.min(5, cachedTestSet.size()); i++) {
            logger.log("TEST[" + i + "]: " + cachedTestSet.get(i));
        }

        return cachedTestSet;
    }

    private int approximateStoreSize() {
        int count = 0;
        try (var it = testStore.all()) {
            while (it.hasNext()) {
                var kv = it.next();
                var vat = kv.value;
                if (vat != null && vat.value() != null) count++;
            }
        }
        return count;
    }

    // private List<DataMessage> loadAllTestDataOnce() {

    //     if (cachedTestSetLoaded && cachedTestSet != null) {
    //         logger.log("Using cached Test Set");
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
    //         logger.log("Could not get assignment for TEST_TOPIC; cannot cache test set.");
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

    //     logger.log("Cached full TEST_TOPIC into memory. Total test rows = " + cachedTestSet.size());
    //     logger.log("First 5 TEST samples:");
    //     for (int i = 0; i < Math.min(5, cachedTestSet.size()); i++) {
    //         logger.log("TEST[" + i + "]: " + cachedTestSet.get(i).toString());
    //     }

    //     return cachedTestSet;
    // }

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
            logger.log("Reached end-of-topic; resetting consumer to beginning (offset 0).");
            return true;
        }

        return false;
    }

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

            logger.log(sb.toString());
        } catch (Exception e) {
            logger.log("Coordinator failed to log consumer offsets: " + e.getMessage());
        }
    }
   
    //=========================================================================================================================

    private static float[] averageWeights(List<float[]> bufs) {
        if (bufs == null || bufs.isEmpty()) return new float[0];

        int numWorkers = bufs.size();
        int len = bufs.get(0).length;
        float[] avg = new float[len];

        for (float[] arr : bufs) {
            for (int i = 0; i < len; i++) {
                avg[i] += arr[i];
            }
        }
        for (int i = 0; i < len; i++) {
            avg[i] /= numWorkers;
        }
        return avg;
    }

    public static class DesiredAccuracyReachedException extends RuntimeException {
        public DesiredAccuracyReachedException(String message) {
            super(message);
        }
    }

}

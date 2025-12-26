package pso;

import org.apache.kafka.streams.processor.api.ContextualProcessor;
import org.apache.kafka.streams.processor.api.Record;
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

import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;

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
    private volatile List<DataMessage> cachedTestSet = null;
    private volatile boolean cachedTestSetLoaded = false;

    // ================================================================================================================

    public CoordinatorProcessor(MultiLayerNetwork globalModel, MultiLayerNetwork bestGlobalModel, long t0, long t1) {
        
        this.t0 = t0;
        this.t1 = t1;

        this.control = CoordinatorControl.getInstance();

        this.logger = CustomLogger.getInstanceForCoordinator();

        this.globalModel = globalModel;
        this.bestGlobalModel = bestGlobalModel;

        this.globalPredictor = BatchPrediction.getInstanceForCoordinator(globalModel, bestGlobalModel, logger);

        Properties consumerProps = new Properties();
        consumerProps.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        consumerProps.put(ConsumerConfig.GROUP_ID_CONFIG, "pso-coordinator-eval-" + RUN_ID);
        consumerProps.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, DataMessageDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest"); // applies only when we dont commit the offset
        consumerProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");
        logger.log("TEST_TOPIC: " + TEST_TOPIC);
        this.consumer = new KafkaConsumer<>(consumerProps);
        this.consumer.subscribe(Collections.singletonList(TEST_TOPIC));    
        
    }

    @Override
    public void init(ProcessorContext<String, WeightsMessage> context) {    // this is output (Kout, Vout)
        this.context = context;
        loadAllTestDataOnce();
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
        
        if ("current_weights".equals(record.key())) {

            // logger.log("Time: " + lastActivitySeconds + " current Position message with msgIndex " + msg.msgIndex + ", from worker " + workerId);
    
            float[] weights = msg.weights;
            if (weights == null) {
                return;
            }
        
            weightsBuffer.put(workerId, weights);

            // Run only if all workers have reported their position 
            if (weightsBuffer.size() == N_WORKERS && cachedTestSetLoaded) { // the particles of the workers should converge so asynchronous communication shouldnt matter
            
                float[] avgWeights = averageWeights(new ArrayList<>(weightsBuffer.values()));

                Dl4jParamUtils.updateModel(globalModel, avgWeights);

                // ======== evaluate accuracy of globalModel using BatchPrediction ========

                // List<DataMessage> evalBatch = new ArrayList<>();

                // while (evalBatch.size() < TEST_SIZE) {  // eval_batch before evaluating performance 

                //     ConsumerRecords<String, DataMessage> records = consumer.poll(Duration.ofMillis(100));

                //     if (records.isEmpty()) {
                //         System.out.println("Test Records run out. Training is over.");
                //         control.requestStop(); // no more test data, training is over
                //         return;
                //     }

                //     for (ConsumerRecord<String, DataMessage> rec : records) {

                //         if (rec.value() != null) {
                //             if(sampledDataMessage == false) {
                //                 logger.log("Sample DataMessage: " + rec.value().toString());
                //                 sampledDataMessage = true;
                //             }
                //             evalBatch.add(rec.value());
                //         }
                //     }
                // }
                
                List<DataMessage> evalBatch;

                if (TEST_SIZE == -1) {
                    evalBatch = loadAllTestDataOnce();
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
                }

                if (TEST_SIZE != -1) {
                    logConsumerOffsets();       // evaluate consumer position
                }

                float[] accLoss = globalPredictor.callPredictionsBatch(evalBatch);
                accuracy = accLoss[0];
                loss = accLoss[1];

                // update bestGlobalModelAccuracy + bestLoss ========================================================

                if(accuracy > bestGlobalModelAccuracy) {    
                    Dl4jParamUtils.updateModel(bestGlobalModel, avgWeights);
                    bestGlobalModelAccuracy = accuracy;
                    logger.log("New bestGlobalModel accuracy = " + bestGlobalModelAccuracy);
                }

                if(loss < bestLoss) {    
                    bestLoss = loss;
                }
                
                updateTime();
                logger.log(test_count + ") time: " + lastActivitySeconds + 
                            ", bestAccuracy: " + bestGlobalModelAccuracy + ", bestLoss: " + bestLoss + 
                            "accuracy: " + accuracy + " and loss: " + loss + 
                            ", and weights sample: " + Dl4jParamUtils.sampleFlatSorted(avgWeights, SAMPLING_CONSTANT));

                System.out.println(test_count + ") time: " + lastActivitySeconds + 
                            ", bestAccuracy: " + bestGlobalModelAccuracy + ", bestLoss: " + bestLoss + 
                            ", model acc: " + accuracy + " and loss: " + loss + 
                            ", and weights sample: " + Dl4jParamUtils.sampleFlatSorted(avgWeights, SAMPLING_CONSTANT));

                test_count++;

                if (bestGlobalModelAccuracy >= this.DESIRED_ACCURACY) {
                    Dl4jParamUtils.saveModel(bestGlobalModel);
                    control.requestStopFinal();
                    return;
                }

                weightsBuffer.clear();
            } 
        
        } else {    // update gBest
            
            logger.log("Time: " + lastActivitySeconds + " pBest message with msgIndex " + msg.msgIndex + ", from worker " + workerId);

            if (msg.loss < gBestLoss) {
                gBestLoss = msg.loss;
                
                WeightsMessage gBestMsg = new WeightsMessage(msg.idWorker, msg.msgIndex, msg.accuracy, msg.loss, msg.weights);

                logger.log("New gBest from worker " + workerId + " with loss: " +  msg.loss + " and with accuracy: " +  msg.accuracy);
                context.forward(new Record<>("gBest", gBestMsg, record.timestamp()));

            }
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

        // Use leftovers from previous poll
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

    private List<DataMessage> loadAllTestDataOnce() {

        if (cachedTestSetLoaded && cachedTestSet != null) {
            logger.log("Using cached Test Set");
            return cachedTestSet;
        }

        consumer.poll(Duration.ZERO);
        Set<TopicPartition> asg = consumer.assignment();

        if (asg == null || asg.isEmpty()) {
            // poll again to get assignment
            consumer.poll(Duration.ofMillis(100));
            asg = consumer.assignment();
        }
        if (asg == null || asg.isEmpty()) {
            logger.log("Could not get assignment for TEST_TOPIC; cannot cache test set.");
            return null;
        }

        consumer.seekToBeginning(asg);
        consumer.poll(Duration.ZERO);

        Map<TopicPartition, Long> ends = consumer.endOffsets(asg);

        List<DataMessage> all = new ArrayList<>(4096);

        while (true) {   // Read until all partitions reach end offsets

            ConsumerRecords<String, DataMessage> records = consumer.poll(Duration.ofMillis(200));

            for (ConsumerRecord<String, DataMessage> rec : records) {
                DataMessage dm = rec.value();
                if (dm != null) all.add(dm);
            }

            boolean allAtEnd = true;
            for (TopicPartition tp : asg) {
                long pos = consumer.position(tp);
                long end = ends.getOrDefault(tp, -1L);

                if (pos < end) {    // if not yet at end
                    allAtEnd = false;
                    break;
                }
            }

            if (allAtEnd) break;    // if at end break
        }

        cachedTestSet = Collections.unmodifiableList(all);
        cachedTestSetLoaded = true;

        logger.log("Cached full TEST_TOPIC into memory. Total test rows = " + cachedTestSet.size());
        logger.log("First 5 TEST samples:");
        for (int i = 0; i < Math.min(5, cachedTestSet.size()); i++) {
            logger.log("TEST[" + i + "]: " + cachedTestSet.get(i).toString());
        }

        return cachedTestSet;
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

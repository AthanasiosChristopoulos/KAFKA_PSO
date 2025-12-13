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
import java.util.Arrays;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.common.serialization.StringDeserializer;

import java.time.Duration;

import utils.*;
import state.*;

public class CoordinatorProcessor implements Processor<String, WeightsMessage, String, WeightsMessage> {
    private ProcessorContext<String, WeightsMessage> context;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final int NUM_WORKERS;

    private final Map<String, float[]> weightsBuffer = new HashMap<>(); // this should be a dictionary of N_WORKER unique "id_worker" keys
    private final Map<String, float[]> pBestBuffer = new HashMap<>(); // this should be a dictionary of N_WORKER unique "id_worker" keys
    
    private float[] gBestWeights = null;
    private float gBestAccuracy = 0f;
    private float gBestLoss = 10000f;

    private int round = 0;

    private final MultiLayerNetwork globalModel; // x_g , current model
    private final MultiLayerNetwork bestGlobalModel; 
    private float bestGlobalModelAccuracy = -1f;

    private final Stats globalStats;
    private final BatchPrediction globalPredictor;

    private final String DATA_TOPIC;
    private final String TEST_TOPIC;

    private final int TEST_SIZE;
    private final float DESIRED_ACCURACY;
    private final String RUN_ID;

    private final KafkaConsumer<String, String> consumer;

    private int count = 0;

    private final CoordinatorControl control;

    private final CustomLogger logger;

    private float accuracy = -1f;
    private float loss = 10000f;

    // ======================================================================

    public CoordinatorProcessor(MultiLayerNetwork globalModel, MultiLayerNetwork bestGlobalModel, Stats globalStats) {

        this.control = CoordinatorControl.getInstance();

        this.logger = CustomLogger.getInstanceForCoordinator();

        this.globalModel = globalModel;
        this.bestGlobalModel = bestGlobalModel;
        this.globalStats = globalStats;    

        this.globalPredictor = BatchPrediction.getInstanceForCoordinator(globalModel, bestGlobalModel, globalStats, logger);

        Config cfg = Config.getInstance();
        this.NUM_WORKERS = cfg.NUM_WORKERS;
        this.TEST_SIZE = cfg.TEST_SIZE;
        this.DATA_TOPIC = cfg.DATA_TOPIC;
        this.TEST_TOPIC = cfg.TEST_TOPIC;
        this.DESIRED_ACCURACY = cfg.DESIRED_ACCURACY;
        this.RUN_ID = cfg.RUN_ID;    
        
        Properties consumerProps = new Properties();
        consumerProps.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        consumerProps.put(ConsumerConfig.GROUP_ID_CONFIG, "pso-coordinator-eval-" + RUN_ID);
        consumerProps.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest"); // applies only when we dont commit the offset
        consumerProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");
        logger.log("TEST_TOPIC: " + TEST_TOPIC);
        this.consumer = new KafkaConsumer<>(consumerProps);
        // this.consumer.subscribe(Collections.singletonList(DATA_TOPIC));     
        this.consumer.subscribe(Collections.singletonList(TEST_TOPIC));     
    }

    @Override
    public void init(ProcessorContext<String, WeightsMessage> context) {    // this is output (Kout, Vout)
        this.context = context;
    }

    @Override
    public void process(Record<String, WeightsMessage> record) {
        if (control.isStopRequested()) return;
        
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

        // logger.log("RECEIVED value: " + value);
        String workerId = String.valueOf(msg.idWorker);

        logger.log("RECEIVED value with msgIndex " + msg.msgIndex + ", from worker " + workerId);

        if ("current_weights".equals(record.key())) {
            
            float[] weights = msg.weights;
            if (weights == null) {
                return;
            }
        
            weightsBuffer.put(workerId, weights);

            // Run only if all workers have reported their position 
            if (weightsBuffer.size() == NUM_WORKERS) { // the particles of the workers should converge so asynchronous communication shouldnt matter
            
                float[] avgWeights = averageWeights(new ArrayList<>(weightsBuffer.values()));

                Dl4jParamUtils.updateModel(globalModel, avgWeights);

                // ======== evaluate accuracy of globalModel using BatchPrediction ========
                List<String> evalBatch = new ArrayList<>();

                while (evalBatch.size() < TEST_SIZE) {  // foll eval_batch before evaluating performance 

                    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(500));

                    if (records.isEmpty()) {
                        System.out.println("Records is empty");
                        break; // no more data, use whatever we have
                    }

                    for (ConsumerRecord<String, String> rec : records) {
                        if (rec.value() != null) {
                            evalBatch.add(rec.value());
                        }
                    }
                }

                globalStats.reset();       
                float[] accLoss = globalPredictor.callPredictionsBatch(evalBatch);
                accuracy = accLoss[0];
                loss = accLoss[1];

                // update bestGlobalModelAccuracy ========================================================
                if(accuracy > bestGlobalModelAccuracy) {    
                    Dl4jParamUtils.updateModel(bestGlobalModel, avgWeights);
                    bestGlobalModelAccuracy = accuracy;
                    logger.log("New bestGlobalModel accuracy = " + bestGlobalModelAccuracy);

                    if (bestGlobalModelAccuracy >= this.DESIRED_ACCURACY) {
                        logger.log("Global model accuracy: " + accuracy + ", bestAccuracy: " + bestGlobalModelAccuracy);
                        System.out.println("Global model accuracy: " + accuracy + ", bestAccuracy: " + bestGlobalModelAccuracy);
                        Dl4jParamUtils.saveModel(bestGlobalModel);
                        control.requestStop();
                        return;
                    }
                }
                
                logger.log("Global model accuracy: " + accuracy + ", bestAccuracy: " + bestGlobalModelAccuracy);
                System.out.println("Global model accuracy: " + accuracy + ", bestAccuracy: " + bestGlobalModelAccuracy);

                weightsBuffer.clear();
            } 
        
        } else {    // update gBest

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

    }

    
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

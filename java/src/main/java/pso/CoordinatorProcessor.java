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

public class CoordinatorProcessor implements Processor<String, WeightsMessage, String, String> {
    private ProcessorContext<String, String> context;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final int NUM_WORKERS;

    private final Map<String, float[]> weightsBuffer = new HashMap<>(); // this should be a dictionary of N_WORKER unique "id_worker" keys
    private final Map<String, float[]> pBestBuffer = new HashMap<>(); // this should be a dictionary of N_WORKER unique "id_worker" keys
    
    private float[] gBestWeights = null;
    private float gBestAccuracy = 0f;

    private int round = 0;

    private final MultiLayerNetwork globalModel; // x_g , current model
    private final Stats globalStats;
    private final BatchPrediction globalPredictor;

    private final String DATA_TOPIC;
    private final int BATCH_SIZE;
    private final float DESIRED_ACCURACY;
    private final String RUN_ID;

    private final KafkaConsumer<String, String> consumer;

    private int count = 0;

    private final CoordinatorControl control;

    private final CustomLogger logger;

    // ======================================================================

    public CoordinatorProcessor(MultiLayerNetwork model, Stats stats) {

        this.control = CoordinatorControl.getInstance();
          
        this.globalModel = model;
        this.globalStats = stats;    

        this.globalPredictor = BatchPrediction.getCoordinatorInstance(globalModel, globalStats);

        Config cfg = Config.getInstance();
        this.NUM_WORKERS = cfg.NUM_WORKERS;
        this.BATCH_SIZE = cfg.BATCH_SIZE;
        this.DATA_TOPIC = cfg.DATA_TOPIC;
        this.DESIRED_ACCURACY = cfg.DESIRED_ACCURACY;
        this.RUN_ID = cfg.RUN_ID;    

        this.logger = CustomLogger.getCoordinatorInstance();
        
        Properties consumerProps = new Properties();
        consumerProps.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        consumerProps.put(ConsumerConfig.GROUP_ID_CONFIG, "pso-coordinator-eval-" + RUN_ID);
        consumerProps.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest"); // applies only when we dont commit the offset
        consumerProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");

        this.consumer = new KafkaConsumer<>(consumerProps);
        this.consumer.subscribe(Collections.singletonList(DATA_TOPIC));     

        logger.log("Coordinator started");
        System.out.println("Coordinator started");

    }

    @Override
    public void init(ProcessorContext<String, String> context) {
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

        logger.log("RECEIVED value with msgIndex " + msg.msgIndex + ", from: " + workerId);

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

            while (evalBatch.size() < BATCH_SIZE) {
                ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(500));

                if (records.isEmpty()) {
                    break; // no more data, use whatever we have
                }

                for (ConsumerRecord<String, String> rec : records) {
                    if (rec.value() != null) {
                        evalBatch.add(rec.value());
                    }
                }
            }

            globalStats.reset();       
            globalPredictor.callPredictionsBatch(evalBatch);
            float accuracy = globalStats.getAccuracy();

            logger.log("Global model accuracy: " + accuracy);
            System.out.println("Global model accuracy: " + accuracy);

            weightsBuffer.clear();

            if (accuracy >= this.DESIRED_ACCURACY) {
                Dl4jParamUtils.saveModel(globalModel);
                control.requestStop();
                return;
            }

        } 
        
        // else {    // Receive pBest updates 

        //     Object pBestObj = msg.get("pBestWeights");
        //     if (!(pBestObj instanceof List<?> pBestList)) {
        //         return;
        //     }

        //     Object accObj = msg.get("accuracy");
        //     if (!(accObj instanceof Number accuracyNumber)) {
        //         logger.log("Received pBest from worker " + workerId + " without numeric accuracy, skipping");
        //         return;
        //     }
        //     float accuracy = accuracyNumber.floatValue();

        //     if (gBestWeights == null || accuracy > gBestAccuracy) {
        //         gBestAccuracy = accuracy;
               
        //         float[] pBestWeights = new float[pBestList.size()];
        //         for (int i = 0; i < pBestList.size(); i++) {
        //             pBestWeights[i] = ((Number) pBestList.get(i)).floatValue();
        //         }

        //         gBestWeights  = pBestWeights;

        //         var payload = new HashMap<String, Object>();
        //         payload.put("id_worker", Integer.parseInt(workerId));
        //         payload.put("pBestMsgIndex", msg.get("pBestMsgIndex"));
        //         payload.put("accuracy", gBestAccuracy);

        //         List<Float> gBestListOut = new ArrayList<>(gBestWeights.length);
        //         for (float v : gBestWeights) {
        //             gBestListOut.add(v);
        //         }
        //         payload.put("gBestWeights", gBestListOut); 

        //         try {
        //             String json = MAPPER.writeValueAsString(payload);
    
        //             logger.log("New gBest from worker " + workerId + " with accuracy " + gBestAccuracy);
        //             logger.log("gBestJSON: " + json);

        //             context.forward(new Record<>(
        //                     "gBest",    // key: all to same partition
        //                     json,       // value: the JSON is the records value 
        //                     record.timestamp()
        //             ));

        //         } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
        //             e.printStackTrace(); // or log it and skip sending
        //         }

        //     }
        // }
    }

    @Override
    public void close() {
        // nothing special
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

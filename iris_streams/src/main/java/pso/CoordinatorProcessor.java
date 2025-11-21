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


public class CoordinatorProcessor implements Processor<String, String, String, String> {
    private ProcessorContext<String, String> context;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final int NUM_WORKERS;

    private final Map<String, double[]> weightsBuffer = new HashMap<>(); // this should be a dictionary of N_WORKER unique "id_worker" keys
    private final Map<String, double[]> pBestBuffer = new HashMap<>(); // this should be a dictionary of N_WORKER unique "id_worker" keys
    
    private double[] gBestWeights = null;
    private double gBestAccuracy = 0.0;

    private int round = 0;

    private final MultiLayerNetwork globalModel;
    private final WorkerStats globalStats;
    private final BatchPrediction globalPredictor;

    private final String DATA_TOPIC;
    private final int BATCH_SIZE;
    private final double DESIRED_ACCURACY;
    private final String SAVE_MODEL_NAME;

    private final KafkaConsumer<String, String> consumer;

    private int count = 0;

    private final CoordinatorControl control;

    // ========================= simple file logger =========================

    private final BufferedWriter logWriter;

    private void log(String msg) {
        if (logWriter == null) return;
        try {
            logWriter.write(msg);
            logWriter.newLine();
            logWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // ======================================================================

    public CoordinatorProcessor(CoordinatorControl control) {

        this.control = control;

        this.NUM_WORKERS = Integer.parseInt(System.getenv().getOrDefault("NUM_WORKERS", "3"));

        this.globalModel = Dl4jModelFactory.createIrisModel();
        this.globalStats = new WorkerStats();                   
        this.globalPredictor = new BatchPrediction(globalModel, globalStats);

        this.DATA_TOPIC = System.getenv().getOrDefault("DATA_TOPIC", "iris-input");
        this.BATCH_SIZE = Integer.parseInt(System.getenv().getOrDefault("BATCH_SIZE", "30"));
        this.DESIRED_ACCURACY = Double.parseDouble(System.getenv().getOrDefault("DESIRED_ACCURACY", "0.9"));
        this.SAVE_MODEL_NAME = System.getenv().getOrDefault("SAVE_MODEL_NAME", "iris-global-model");

        BufferedWriter w = null;
        try {
            Files.createDirectories(Paths.get("logs"));
            w = Files.newBufferedWriter(
                    Paths.get("logs/coordinator.log"),
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            );
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.logWriter = w;
        
        String RUN_ID = System.getenv().getOrDefault("RUN_ID", "111");

        Properties consumerProps = new Properties();
        consumerProps.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        consumerProps.put(ConsumerConfig.GROUP_ID_CONFIG, "pso-coordinator-eval-" + RUN_ID);
        consumerProps.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        consumerProps.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest"); // applies only when we dont commit the offset
        consumerProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");

        this.consumer = new KafkaConsumer<>(consumerProps);
        this.consumer.subscribe(Collections.singletonList(DATA_TOPIC));     

        log("Coordinator started");
        System.out.println("Coordinator started");

    }

    @Override
    public void init(ProcessorContext<String, String> context) {
        this.context = context;
    }

    @Override
    public void process(Record<String, String> record) {
        if (control.isStopRequested()) return;
        
        if(count == 0) {
            context.recordMetadata().ifPresent(meta -> 
                log("Starting Meta Data: " + meta.topic() + ", Partition: " + meta.partition() + ", Offset: " + meta.offset())
            );
        }

        count = count + 1;

        String value = record.value();
        if (value == null) {
            return;
        }

        Map<String, Object> msg;
        try {
            msg = MAPPER.readValue(value, new TypeReference<Map<String, Object>>() {});
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            e.printStackTrace(); 
            return;
        }

        Object workerIdObj = msg.get("id_worker");
        if (workerIdObj == null) {
            return;
        }
        // log("Unexpected worker_id type: " + workerIdObj.getClass());
        // int workerId = ((Number) msg.get("id_worker")).intValue();

        String workerId = String.valueOf(workerIdObj);

        log("RECEIVED value: " + value);

        if(msg.containsKey("weights")) {    // Receive weight updates
            
            Object weightsObj = msg.get("weights");
            if (!(weightsObj instanceof List<?> weightsList)) { // if weightsObj is a List,then automatically cast it into weightsList
                return;
            }

            double[] weights = new double[weightsList.size()];
            for (int i = 0; i < weightsList.size(); i++) {
                weights[i] = ((Number) weightsList.get(i)).doubleValue();
            }

            weightsBuffer.put(workerId, weights);

            // Run only if all workers have reported their position 
            if (weightsBuffer.size() == NUM_WORKERS) { // the particles of the workers should converge so asynchronous communication shouldnt matter
            
                double[] avgWeights = averageWeights(new ArrayList<>(weightsBuffer.values()));

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
                double accuracy = globalStats.getAccuracy();

                log("Global model accuracy: " + accuracy);
                System.out.println("Global model accuracy: " + accuracy);

                weightsBuffer.clear();

               if (accuracy >= this.DESIRED_ACCURACY) {
                    Dl4jParamUtils.saveModel(globalModel, SAVE_MODEL_NAME);
                    control.requestStop();
                    return;
                }
            }

        } else {    // Receive pBest updates 

            Object pBestObj = msg.get("pBest");
            if (!(pBestObj instanceof List<?> pBestList)) {
                return;
            }

            Object accObj = msg.get("accuracy");
            if (!(accObj instanceof Number accuracyNumber)) {
                log("Received pBest from worker " + workerId + " without numeric accuracy, skipping");
                return;
            }
            double accuracy = accuracyNumber.doubleValue();

            // --- check against current gBest ---
            if (gBestWeights == null || accuracy > gBestAccuracy) {
                // new global best!
                gBestAccuracy = accuracy;
               
                double[] pBestWeights = new double[pBestList.size()];
                for (int i = 0; i < pBestList.size(); i++) {
                    pBestWeights[i] = ((Number) pBestList.get(i)).doubleValue();
                }

                gBestWeights  = pBestWeights;

                var payload = new HashMap<String, Object>();
                payload.put("id_worker", Integer.parseInt(workerId));
                payload.put("pBestMsgIndex", msg.get("pBestMsgIndex"));
                payload.put("accuracy", gBestAccuracy);

                List<Double> gBestListOut = new ArrayList<>(gBestWeights.length);
                for (double v : gBestWeights) {
                    gBestListOut.add(v);
                }
                payload.put("w_gBest", gBestListOut); 

                try {
                    String json = MAPPER.writeValueAsString(payload);
    
                    log("New gBest from worker " + workerId + " with accuracy " + gBestAccuracy);
                    log("gBestJSON: " + json);

                    context.forward(new Record<>(
                            "gBest",    // key: all to same partition
                            json,       // value: the JSON is the records value 
                            record.timestamp()
                    ));

                } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
                    e.printStackTrace(); // or log it and skip sending
                }

            }
        }
    }

    @Override
    public void close() {
        // nothing special
    }

    
    private static double[] averageWeights(List<double[]> bufs) {
        if (bufs == null || bufs.isEmpty()) return new double[0];

        int numWorkers = bufs.size();
        int len = bufs.get(0).length;
        double[] avg = new double[len];

        for (double[] arr : bufs) {
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

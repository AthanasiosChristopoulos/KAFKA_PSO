
package pso;

import org.apache.kafka.streams.processor.api.ContextualProcessor;
import org.apache.kafka.streams.processor.api.Record;
import org.apache.kafka.streams.processor.api.ProcessorContext;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.BufferedWriter;
import java.nio.file.StandardOpenOption;

import org.json.JSONObject;
import org.json.JSONArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;
import com.fasterxml.jackson.core.type.TypeReference;


public class PredictionBatchProcessor extends ContextualProcessor<String, String, String, String> {

    private final List<String> buffer = new ArrayList<>();
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final PsoUpdater psoUpdater;
    private int batchesRead = 0;

    private final MultiLayerNetwork model;
    private final WorkerStats stats;
    private final BatchPrediction predictor;

    private final int BATCH_SIZE;
    private final int N_BATCHES;

    private int count = 0;
    private int pBestMsgIndex;
    private int weightsMsgIndex;

    private int workerId;

    private double[] pBestWeights = null;

    private WorkerSharedState shared;

    // ======================================================================

    public PredictionBatchProcessor(int workerId, WorkerSharedState shared) {

        this.workerId = workerId;

        model = Dl4jModelFactory.createIrisModel();
            
        this.stats = new WorkerStats();
        this.predictor = new BatchPrediction(model, stats);

        this.BATCH_SIZE = Integer.parseInt(System.getenv().getOrDefault("BATCH_SIZE", "30"));
        this.N_BATCHES = Integer.parseInt(System.getenv().getOrDefault("N_BATCHES", "10"));

        this.psoUpdater = new PsoUpdater(model, workerId);
        
        this.shared = shared;

        this.pBestWeights = Dl4jParamUtils.modelToFlatList(model);  // just an initialization this will get updated soon
        shared.gBestWeights = Dl4jParamUtils.modelToFlatList(model); 

        shared.log("Worker " + workerId + " started");

    }

    //============================================================================================

    @Override
    public void init(ProcessorContext<String, String> context) {
        super.init(context);
    }

    @Override
    public void process(Record<String, String> record) {

        if(count == 0) {
            context().recordMetadata().ifPresent(meta -> 
                shared.log("Starting Meta Data: " + meta.topic() + ", Partition: " + meta.partition() + ", Offset: " + meta.offset())
            );
        }

        count = count + 1;
        // shared.log("New Record: " + count);  

        String value = record.value();
        if (value == null) {
            return;
        }
        // shared.log("VALUE = " + value);
        // Map<String, Object> msg = MAPPER.readValue(value, new TypeReference<Map<String, Object>>() {});

        Map<String, Object> msg;
        try {
            msg = MAPPER.readValue(value, new TypeReference<Map<String, Object>>() {});
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            shared.log("Something went wrong when parsing the JSON");
            e.printStackTrace();
            return;
        }

        // shared.log("ddd = " + msg.containsKey("id_worker"));

        if(msg.containsKey("id_worker")) { // if the record contains a worker_id, Receive weights

            shared.log("RECEIVED global gBestWeights: " + value);
            Object workerIdObj = msg.get("id_worker");
            if (workerIdObj == null) {
                return;
            }
            String workerId = String.valueOf(workerIdObj);


            Object gBestObj = msg.get("w_gBest");
            if (!(gBestObj instanceof List<?> gBestList)) {
                return;
            }

            double[] gBestWeights = new double[gBestList.size()];
            for (int i = 0; i < gBestList.size(); i++) {
                gBestWeights[i] = ((Number) gBestList.get(i)).doubleValue();
            }

            shared.gBestWeights = gBestWeights; // should be executed only in the start
            shared.gBestWeightsChanged = 1;

            shared.log("Current gBestWeights: \n" + Arrays.toString(shared.gBestWeights));

        //============================================================================================================

        } else { // Receive Data

            // shared.log("VALUE = " + value);
            buffer.add(value); // accumulate records until you reach batch size

            if (buffer.size() >= BATCH_SIZE) {

                predictor.callPredictionsBatch(buffer);
                buffer.clear(); // processed the buffer, now empty it

                batchesRead = batchesRead + 1;

                double accuracy = stats.getAccuracy();

                if(accuracy == -1) {
                    return;
                }

                double[] weights = Dl4jParamUtils.modelToFlatList(model);
                java.util.List<Double> weightList = new java.util.ArrayList<>(weights.length);
                for (double v : weights) {
                    weightList.add(v);
                }

                if(accuracy > stats.getBestAccuracy()) {

                    stats.setBestAccuracy(accuracy);
                    this.pBestWeights = weights;

                    shared.log("Improved accuracy to: " + stats.getBestAccuracy() + ", n_predictions: " +
                                stats.getNumPredictions() + ", n_correct: " + stats.getNumCorrect());

                    var payload = new java.util.HashMap<String, Object>();
                    payload.put("pBestMsgIndex", java.util.UUID.randomUUID().toString());
                    payload.put("id_worker", workerId);  
                    payload.put("pBest", weightList);   // update pBest
                    payload.put("accuracy", accuracy);  // accuracy of that new model

                    try {
                        String json = MAPPER.writeValueAsString(payload);
                        shared.log("SENDING pBest JSON: " + json); 
                        context().forward(new Record<>(
                            String.valueOf(1),  // this is the key (irrelevant here)
                                                // all forwarded messages go to the same partition, because the key is always "1".
                            json,               // this is the value of the record
                            record.timestamp()
                        ));

                    } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
                        e.printStackTrace(); // or shared.log it and skip sending
                    }

                }

                if (batchesRead >= N_BATCHES) { // send x_i (current position). Dont send all the time only after processing many batches.
                    // context().recordMetadata().ifPresent(meta -> // log offset for input topic
                    //     shared.log("Current Meta Data: " + meta.topic() + ", Partition: " + meta.partition() + ", Offset: " + meta.offset())
                    // );
                    try {

                        var payload = new java.util.HashMap<String, Object>();
                        payload.put("id_worker", this.workerId);  
                        payload.put("weightsMsgIndex", java.util.UUID.randomUUID().toString());
                        payload.put("weights", weightList);

                        String json = MAPPER.writeValueAsString(payload);

                        context().forward(new Record<>(
                            String.valueOf(1),   // key
                            json,                // value
                            record.timestamp()   
                        ));

                        batchesRead = 0;

                        // shared.log("Uploaded weights after " + N_BATCHES + " batches");

                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

                double[] velocity = psoUpdater.updateX(model, this.pBestWeights, shared.gBestWeights);
                stats.reset();
                shared.log("Updated Model to: " + Dl4jParamUtils.sampleFlat(Dl4jParamUtils.modelToFlatList(model)) 
                            + ", with velocity: "+ Dl4jParamUtils.sampleFlat(velocity));

            }
        }
    }

    @Override
    public void close() {
        if (!buffer.isEmpty()) {
            predictor.callPredictionsBatch(buffer);
            buffer.clear();
        }
    }
}



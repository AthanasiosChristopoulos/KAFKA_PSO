package pso;

import org.apache.kafka.streams.kstream.Transformer;
import org.apache.kafka.streams.KeyValue;
import org.apache.kafka.streams.processor.ProcessorContext;

import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.streams.state.ReadOnlyKeyValueStore;
import org.apache.kafka.streams.state.ValueAndTimestamp;
import org.apache.kafka.streams.state.KeyValueIterator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;

import java.util.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;

import utils.*;
import state.*;

public class WorkerTransformer implements Transformer<String, String, KeyValue<String, String>> {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final int workerId;
    private final int BATCH_SIZE;
    private final int N_BATCHES;
    private final String FULLY_INFORMED;

    private final CustomLogger logger;

    private ProcessorContext context;

    private ReadOnlyKeyValueStore<String, ValueAndTimestamp<String>> bestStore;

    private final List<String> buffer = new ArrayList<>();

    private final MultiLayerNetwork model;
    private final Stats stats;
    private final BatchPrediction predictor;
    private final PsoUpdater psoUpdater;

    private double[] pBestWeights;
    private int batchesRead = 0;

    private boolean printedOffset = false;

    private String stateStoreName;
    private String keyName;

    public WorkerTransformer(int workerId) {

        this.workerId = workerId;
        
        Config cfg = Config.get();
        this.BATCH_SIZE = cfg.BATCH_SIZE;
        this.N_BATCHES = cfg.N_BATCHES;   
        this.FULLY_INFORMED = cfg.FULLY_INFORMED;

        this.model = Dl4jModelFactory.createIrisModel();
        this.stats = new Stats();
        this.predictor = new BatchPrediction(model, stats);
        this.psoUpdater = new PsoUpdater(model, workerId);

        this.pBestWeights = Dl4jParamUtils.modelToFlatList(model);

        this.logger = CustomLogger.getWorkerInstance(workerId);
        logger.log("Worker " + workerId + " WorkerTransformer started");

        if("true".equals(FULLY_INFORMED)) {
            stateStoreName = "pBestStore";
            keyName = "pBest" + workerId;
        } else {
            stateStoreName = "gBestStore";
            keyName = "gBest";
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public void init(ProcessorContext context) {
        this.context = context;
        this.bestStore = (ReadOnlyKeyValueStore<String, ValueAndTimestamp<String>>) context.getStateStore(stateStoreName);
        // bestStore = null;
    }

    //=========================================================================================================================

    @Override
    public KeyValue<String, String> transform(String key, String value) {

        if (!printedOffset) {
            printedOffset = true;
            logger.log(
                "Starting at -> " +
                "Offset: " + context.offset() +
                ", Partition: " + context.partition() +
                ", Topic: " + context.topic()
            );
            if (bestStore == null) {
                logger.log("gBestWeights returned null");
                return null;
            }

            ValueAndTimestamp<String> wrapper = bestStore.get(keyName);
            if (wrapper == null) {
                logger.log("initial bestStore: bestStore has no 'gBest'");
            } else {
                logger.log("initial bestStore: bestStore['gBest'] = " + wrapper.value());
            }
        }
        
        if (value == null) {
            return null;
        }

        buffer.add(value);

        if (buffer.size() < BATCH_SIZE) {
            return null;
        }

        predictor.callPredictionsBatch(buffer);
        buffer.clear();
        batchesRead++;

        double accuracy = stats.getAccuracy();
        if (accuracy == -1) {
            return null;
        }

        double[] weights = Dl4jParamUtils.modelToFlatList(model);
        List<Double> weightList = new ArrayList<>(weights.length);
        for (double v : weights) {
            weightList.add(v);
        }

        if (accuracy > stats.getBestAccuracy()) {

            stats.setBestAccuracy(accuracy);
            this.pBestWeights = weights;

            logger.log("Improved accuracy to: " + stats.getBestAccuracy() + ", n_predictions: " + stats.getNumPredictions() +
                       ", n_correct: " + stats.getNumCorrect());

            var payload = new HashMap<String, Object>();
            payload.put("pBestMsgIndex", java.util.UUID.randomUUID().toString());
            payload.put("id_worker", workerId);
            payload.put("pBestWeights", weightList);
            payload.put("accuracy", accuracy);

            try {
                String json = MAPPER.writeValueAsString(payload);
                // logger.log("SENDING pBest JSON: " + json);

                return new KeyValue<>(keyName, json);

            } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
                e.printStackTrace();
                return null;
            }
        }

        if (batchesRead >= N_BATCHES) {
            try {
                logger.log("Sending current weights ...");
                var payload = new HashMap<String, Object>();
                payload.put("id_worker", this.workerId);
                payload.put("weightsMsgIndex", java.util.UUID.randomUUID().toString());
                payload.put("weights", weightList);

                String json = MAPPER.writeValueAsString(payload);

                batchesRead = 0;

                return new KeyValue<>("1", json);

            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        }

        double[] velocity = new double[this.pBestWeights.length];

        // get the State Store ==================================================================================

        if ("true".equals(FULLY_INFORMED)) {

            List<double[]> neighborPBestList = readNeighborPBestList();

            if (neighborPBestList == null || neighborPBestList.isEmpty()) {
                logger.log("No neighbor pBest found; skipping social update this round.");
                velocity = psoUpdater.updateX(model, null);
            } else {
                logger.log("pBest Weights:\n" + Dl4jParamUtils.sampleFlats(neighborPBestList));
                velocity = psoUpdater.updateX(model, neighborPBestList);
            }

        } else {

            double[] gBestWeights = readBestWeights();
            if (gBestWeights == null) {
                gBestWeights = new double[this.pBestWeights.length];
            }
            // logger.log("gBest Weight: " + Dl4jParamUtils.sampleFlat(gBestWeights));
            velocity = psoUpdater.updateX(model, this.pBestWeights, gBestWeights);
        }

        stats.reset();
        logger.log("Updated Model to: " + Dl4jParamUtils.sampleFlat(Dl4jParamUtils.modelToFlatList(model)) +
                    ", with velocity: " + Dl4jParamUtils.sampleFlat(velocity));

        return null;
    }

    //=========================================================================================================================

    private List<double[]> readNeighborPBestList() {

        // dumpBestStore();

        List<double[]> neighbors = new ArrayList<>();

        if (bestStore == null) {
            logger.log("readNeighborPBestList: bestStore is null");
            return neighbors;
        }

        try (KeyValueIterator<String, ValueAndTimestamp<String>> it = bestStore.all()) {

            while (it.hasNext()) {
                KeyValue<String, ValueAndTimestamp<String>> entry = it.next();

                String json = entry.value.value();
                if (json == null) continue;

                try {
                    Map<String, Object> msg = MAPPER.readValue(json, new TypeReference<Map<String, Object>>() {});
                    Object pBestObj = msg.get("pBestWeights");
                    if (!(pBestObj instanceof List<?> pBestList)) {
                        continue;
                    }

                    double[] pBestArr = new double[pBestList.size()];
                    for (int i = 0; i < pBestList.size(); i++) {
                        pBestArr[i] = ((Number) pBestList.get(i)).doubleValue();
                    }

                    neighbors.add(pBestArr);

                } catch (Exception e) {
                    logger.log("Error parsing pBest from store: " + e.getMessage());
                    e.printStackTrace();
                }
            }

        } catch (Exception e) {
            logger.log("Error iterating bestStore: " + e.getMessage());
            e.printStackTrace();
        }

        return neighbors;
    }

    //=========================================================================================================================

    private double[] readBestWeights() { // read state store

        if (bestStore == null) {
            logger.log("gBestWeights returned null");
            return null;
        }

        // String gBestJson = bestStore.get(keyName); // this is the State Store. get(record key)
        // dumpBestStore();

        ValueAndTimestamp<String> wrapper = bestStore.get(keyName);
        if (wrapper == null) {
            logger.log("gBestWeights returned null (no entry for key 'gBest')");
            return null;
        }

        String gBestJson = wrapper.value();
        logger.log("gBestJson: " + gBestJson);

        try {

            Map<String, Object> msg = MAPPER.readValue(gBestJson, new TypeReference<Map<String, Object>>() {});
            Object gBestObj = msg.get("gBestWeights"); // get MSG inside the JSON
            if (!(gBestObj instanceof List<?> gBestList)) {
                logger.log("gBestWeights isnt a List");
                return null;
            }

            double[] gBestWeights = new double[gBestList.size()]; // initialize with 0.0 values

            for (int i = 0; i < gBestList.size(); i++) {
                gBestWeights[i] = ((Number) gBestList.get(i)).doubleValue(); // fill the gBestWeights with the actuall values
            }
            // logger.log("gBestWeights: " + Dl4JParamUtils.sampleFlat(gBestWeights));

            return gBestWeights;

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    // =====================================================================================================================

    private void dumpBestStore() {

        try (KeyValueIterator<String, ValueAndTimestamp<String>> it = bestStore.all()) {

            boolean empty = true;

            while (it.hasNext()) {
                empty = false;
                KeyValue<String, ValueAndTimestamp<String>> entry = it.next();

                logger.log(
                    "[bestStore] key = " + entry.key +
                    ", value = " + entry.value.value() +
                    ", timestamp = " + entry.value.timestamp()
                );
            }

            if (empty) {
                logger.log("[bestStore] Store is empty!");
            }

        } catch (Exception e) {
            logger.log("Error while dumping bestStore: " + e.getMessage());
            e.printStackTrace();
        }
    }

    //=========================================================================================================================

    @Override
    public void close() {
        if (!buffer.isEmpty()) {
            buffer.clear();
        }
    }
}

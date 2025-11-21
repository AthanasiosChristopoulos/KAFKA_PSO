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
    private final int batchSize;
    private final int nBatches;
    private final CustomLogger logger;

    private ProcessorContext context;
    // private KeyValueStore<Strin  g, String> gBestStore;  // <-- state store
    // private ReadOnlyKeyValueStore<String, String> gBestStore;

    private ReadOnlyKeyValueStore<String, ValueAndTimestamp<String>> gBestStore;

    private final List<String> buffer = new ArrayList<>();

    private final MultiLayerNetwork model;
    private final Stats stats;
    private final BatchPrediction predictor;
    private final PsoUpdater psoUpdater;

    private double[] pBestWeights;
    private int batchesRead = 0;

    private boolean printedOffset = false;

    public WorkerTransformer(int workerId, int batchSize, int nBatches) {

        this.workerId = workerId;
        this.batchSize = batchSize;
        this.nBatches = nBatches;

        this.model = Dl4jModelFactory.createIrisModel();
        this.stats = new Stats();
        this.predictor = new BatchPrediction(model, stats);
        this.psoUpdater = new PsoUpdater(model, workerId);

        this.pBestWeights = Dl4jParamUtils.modelToFlatList(model);

        this.logger = new CustomLogger(workerId);
        logger.log("Worker " + workerId + " WorkerTransformer started");
    }

    @Override
    @SuppressWarnings("unchecked")
    public void init(ProcessorContext context) {
        this.context = context;
        // this.gBestStore = (KeyValueStore<String, String>) context.getStateStore("gBestStore"); // open state store
        // this.gBestStore = (ReadOnlyKeyValueStore<String, String>) context.getStateStore("gBestStore");
        this.gBestStore = (ReadOnlyKeyValueStore<String, ValueAndTimestamp<String>>) context.getStateStore("gBestStore");
    }

    //=========================================================================================================================

    @Override
    public KeyValue<String, String> transform(String key, String value) {

        if (!printedOffset) {
            logger.log(
                "Starting at -> " +
                "Offset: " + context.offset() +
                ", Partition: " + context.partition() +
                ", Topic: " + context.topic()
            );
            ValueAndTimestamp<String> wrapper = gBestStore.get("gBest");
            if (wrapper == null) {
                logger.log("initial gBestStore: gBestStore has no 'gBest'");
            } else {
                logger.log("initial gBestStore: gBestStore['gBest'] = " + wrapper.value());
            }
            
            printedOffset = true;
        }

        
        if (value == null) {
            return null;
        }

        buffer.add(value);

        if (buffer.size() < batchSize) {
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
            payload.put("pBest", weightList);
            payload.put("accuracy", accuracy);

            try {
                String json = MAPPER.writeValueAsString(payload);
                logger.log("SENDING pBest JSON: " + json);

                return new KeyValue<>("gBest", json);

            } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
                e.printStackTrace();
                return null;
            }
        }

        if (batchesRead >= nBatches) {
            try {
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

        double[] gBestWeights = readGlobalBestWeights();

        if (gBestWeights == null) {
            gBestWeights = new double[this.pBestWeights.length]; // make a 0.0 array, essentially making this parameter ineffective
        }

        double[] velocity = psoUpdater.updateX(model, this.pBestWeights, gBestWeights);
        stats.reset();

        logger.log("Updated Model to: " + Dl4jParamUtils.sampleFlat(Dl4jParamUtils.modelToFlatList(model)) +
                    ", with velocity: " + Dl4jParamUtils.sampleFlat(velocity));

        return null;
    }

    //=========================================================================================================================

    private double[] readGlobalBestWeights() { // read state store

        if (gBestStore == null) {
            logger.log("gBestWeights returned null");
            return null;
        }

        // String gBestJson = gBestStore.get("gBest"); // this is the State Store. get(record key)
        dumpGbestStore();

        ValueAndTimestamp<String> wrapper = gBestStore.get("gBest");
        if (wrapper == null) {
            logger.log("gBestWeights returned null (no entry for key 'gBest')");
            return null;
        }

        String gBestJson = wrapper.value();
        logger.log("gBestJson: " + gBestJson);
        if (gBestJson == null) {
            logger.log("gBestWeights returned null");
            return null;
        }

        try {

            Map<String, Object> msg = MAPPER.readValue(gBestJson, new TypeReference<Map<String, Object>>() {});
            Object gBestObj = msg.get("w_gBest"); // get MSG inside the JSON
            if (!(gBestObj instanceof List<?> gBestList)) {
                logger.log("gBestWeights isnt a List");
                return null;
            }

            double[] gBestWeights = new double[gBestList.size()]; // initialize with 0.0 values

            for (int i = 0; i < gBestList.size(); i++) {
                gBestWeights[i] = ((Number) gBestList.get(i)).doubleValue(); // fill the gBestWeights with the actuall values
            }
            logger.log("gBestWeights: " + Arrays.toString(gBestWeights));

            return gBestWeights;

        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    private void dumpGbestStore() {

        if (gBestStore == null) {
            logger.log("gBestStore is null (not initialized yet)");
            return;
        }

        try (KeyValueIterator<String, ValueAndTimestamp<String>> it = gBestStore.all()) {

            boolean empty = true;

            while (it.hasNext()) {
                empty = false;
                KeyValue<String, ValueAndTimestamp<String>> entry = it.next();

                logger.log(
                    "[gBestStore] key = " + entry.key +
                    ", value = " + entry.value.value() +
                    ", timestamp = " + entry.value.timestamp()
                );
            }

            if (empty) {
                logger.log("[gBestStore] Store is empty!");
            }

        } catch (Exception e) {
            logger.log("Error while dumping gBestStore: " + e.getMessage());
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

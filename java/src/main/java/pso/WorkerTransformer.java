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

public class WorkerTransformer implements Transformer<String, String, KeyValue<String, WeightsMessage>> {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final int workerId;
    private final int BATCH_SIZE;
    private final int N_BATCHES;
    private final boolean FULLY_INFORMED;

    private final CustomLogger logger;

    private ProcessorContext context;

    private ReadOnlyKeyValueStore<String, ValueAndTimestamp<WeightsMessage>> bestStore;

    private final List<String> buffer = new ArrayList<>();

    private final MultiLayerNetwork model;
    private final Stats stats;
    private final BatchPrediction predictor;
    private final PsoUpdater psoUpdater;

    private float[] pBestWeights;
    private int batchesRead = 0;

    private boolean printedOffset = false;

    private String stateStoreName;
    private String keyName;

    public WorkerTransformer(int workerId) {

        this.workerId = workerId;
        
        Config cfg = Config.getInstance();
        this.BATCH_SIZE = cfg.BATCH_SIZE;
        this.N_BATCHES = cfg.N_BATCHES;   
        this.FULLY_INFORMED = cfg.FULLY_INFORMED;

        this.model = Dl4jModelFactory.createModel();
        this.stats = new Stats();
        this.predictor = new BatchPrediction(model, stats);
        this.psoUpdater = new PsoUpdater(model, workerId);

        this.pBestWeights = Dl4jParamUtils.modelToFlatList(model);

        this.logger = CustomLogger.getWorkerInstance(workerId);
        logger.log("Worker " + workerId + " WorkerTransformer started");

        if(FULLY_INFORMED == true) {
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
        this.bestStore = (ReadOnlyKeyValueStore<String, ValueAndTimestamp<WeightsMessage>>) context.getStateStore(stateStoreName);
    }

    //=========================================================================================================================

    @Override
    public KeyValue<String, WeightsMessage> transform(String key, String value) {

        if (!printedOffset) {
            printedOffset = true;
            logger.log(
                "Starting at -> " +
                "Offset: " + context.offset() +
                ", Partition: " + context.partition() +
                ", Topic: " + context.topic()
            );
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

        float accuracy = stats.getAccuracy();
        float loss = stats.getLoss();
        if (accuracy == 0.0) {
            System.out.println("Accuracy Invalid");
            return null;
        }

        float[] weights = Dl4jParamUtils.modelToFlatList(model);

        if(loss < stats.getBestLoss()) {

            stats.setBestAccuracy(accuracy);
            stats.setBestLoss(loss);

            this.pBestWeights = weights;

            String msgIndex = java.util.UUID.randomUUID().toString();

            logger.log("Improved loss: " + stats.getBestLoss() + " and accuracy: " + stats.getBestAccuracy() +
                        ", msgIndex = " + msgIndex +
                        ", n_predictions: " + stats.getNumPredictions() + ", n_correct: " + stats.getNumCorrect());

            WeightsMessage msg = new WeightsMessage(workerId, msgIndex, accuracy, loss, weights);

            return new KeyValue<>(keyName, msg);
        }

        if (batchesRead >= N_BATCHES) {

            logger.log("Sending current weights ...");

            batchesRead = 0;
            String msgIndex = java.util.UUID.randomUUID().toString();

            WeightsMessage msg = new WeightsMessage(workerId, msgIndex, accuracy, loss, weights);

            return new KeyValue<>("current_weights", msg);

        }

        float[] velocity = new float[this.pBestWeights.length];

        // get the State Store ==================================================================================

        if (FULLY_INFORMED == true) {

            List<float[]> neighborPBestList = readNeighborPBestList();

            if (neighborPBestList == null || neighborPBestList.isEmpty()) {
                logger.log("No neighbor pBest found; skipping social update this round.");
                velocity = psoUpdater.updateX(model, null);
            } else {
                logger.log("pBest Weights:\n" + Dl4jParamUtils.sampleFlats(neighborPBestList));
                velocity = psoUpdater.updateX(model, neighborPBestList);
            }

        } else {

            float[] gBestWeights = readBestWeights();
            if (gBestWeights == null) {
                gBestWeights = new float[this.pBestWeights.length];
            } else {
                logger.log("gBest Weight: " + Dl4jParamUtils.sampleFlat(gBestWeights));
            }

            velocity = psoUpdater.updateX(model, this.pBestWeights, gBestWeights);
        }

        stats.reset();
        logger.log("Updated Model to: " + Dl4jParamUtils.sampleFlat(Dl4jParamUtils.modelToFlatList(model)) +
                    ", with velocity: " + Dl4jParamUtils.sampleFlat(velocity) + 
                    ", with loss: " + loss + ", with accuracy: " + accuracy);

        return null;
    }

    //=========================================================================================================================

    private List<float[]> readNeighborPBestList() {

        List<float[]> neighbors = new ArrayList<>();

        if (bestStore == null) {
            logger.log("readNeighborPBestList: bestStore is null");
            return neighbors;
        }

        try (KeyValueIterator<String, ValueAndTimestamp<WeightsMessage>> it = bestStore.all()) {

            while (it.hasNext()) {
                KeyValue<String, ValueAndTimestamp<WeightsMessage>> entry = it.next();
                WeightsMessage msg = entry.value.value(); 
                if (msg == null || msg.weights == null || msg.weights.length == 0) {
                    continue;
                }

                float[] pBestArr = msg.weights;

                neighbors.add(pBestArr);
            }

        } catch (Exception e) {
            logger.log("Error iterating bestStore: " + e.getMessage());
            e.printStackTrace();
        }

        return neighbors;
    }


    //=========================================================================================================================

    private float[] readBestWeights() {

        if (bestStore == null) {
            logger.log("bestStore is null");
            return null;
        }

        ValueAndTimestamp<WeightsMessage> wrapper = bestStore.get(keyName);
        if (wrapper == null) {
            logger.log("gBestWeights returned null (no entry for key '" + keyName + "')");
            return null;
        }

        WeightsMessage best = wrapper.value();
        if (best == null || best.weights == null || best.weights.length == 0) {
            logger.log("gBestWeights is empty for key '" + keyName + "'");
            return null;
        }

        float[] gBestWeights = best.weights;
        if (gBestWeights == null || gBestWeights.length == 0) {
            logger.log("gBestWeights is empty for key '" + keyName + "'");
            return null;
        }

        // logger.log("gBestWeights: " + Dl4jParamUtils.sampleFlat(gBestWeights));
        return gBestWeights;
    }

    // =====================================================================================================================

    // private void dumpBestStore() {

    //     if (bestStore == null) {
    //         logger.log("[bestStore] Store is null!");
    //         return;
    //     }

    //     try (KeyValueIterator<String, ValueAndTimestamp<WeightsMessage>> it = bestStore.all()) {

    //         boolean empty = true;

    //         while (it.hasNext()) {
    //             empty = false;
    //             KeyValue<String, ValueAndTimestamp<WeightsMessage>> entry = it.next();

    //             WeightsMessage msg = entry.value.value(); // unwrap
    //             if (msg == null) {
    //                 logger.log("[bestStore] key = " + entry.key + ", value = null");
    //                 continue;
    //             }

    //             int nWeights = (msg.weights != null) ? msg.weights.length : 0;

    //             logger.log(
    //                 "[bestStore] key = " + entry.key +
    //                 ", id_worker = " + msg.idWorker +
    //                 ", msgIndex = " + msg.msgIndex +
    //                 ", accuracy = " + msg.accuracy +
    //                 ", loss = " + msg.loss +
    //                 ", nWeights = " + nWeights
    //             );
    //         }

    //         if (empty) {
    //             logger.log("[bestStore] Store is empty!");
    //         }

    //     } catch (Exception e) {
    //         logger.log("Error while dumping bestStore: " + e.getMessage());
    //         e.printStackTrace();
    //     }
    // }



    //=========================================================================================================================

    @Override
    public void close() {
        if (!buffer.isEmpty()) {
            buffer.clear();
        }
    }
}

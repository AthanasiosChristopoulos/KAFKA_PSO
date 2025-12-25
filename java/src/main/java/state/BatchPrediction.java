
package state;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;

import utils.*;
import message.data_message.*; 
import message.weights_message.*; 

public class BatchPrediction {

    private static final Config cfg = Config.getInstance();
    public final int NUM_FEATURES = cfg.NUM_FEATURES;
    public final int NUM_CLASSES = cfg.NUM_CLASSES;
    public final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;

    public final String DATASET = cfg.DATASET;
    private static final String LOSS_FUNCTION = cfg.LOSS_FUNCTION;
     private static final String LOSS_COMBINE = cfg.LOSS_COMBINE;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final MultiLayerNetwork model;
    private final MultiLayerNetwork bestModel;

    private static BatchPrediction coordinatorInstance = null;

    private final CustomLogger logger;

    // for Worker =======================================================================================================

    public BatchPrediction(MultiLayerNetwork model, CustomLogger logger) {
        this.model = model;
        this.bestModel = null;
        this.logger = logger;
    }

    // for Coordinator ==================================================================================================

    public BatchPrediction(MultiLayerNetwork model, MultiLayerNetwork bestModel, CustomLogger logger) {
        this.model = model;
        this.bestModel = bestModel;
        this.logger = logger;
    }

    public static BatchPrediction getInstanceForCoordinator(MultiLayerNetwork model, MultiLayerNetwork bestModel, CustomLogger logger) {
        if(coordinatorInstance == null) {
            coordinatorInstance = new BatchPrediction(model, bestModel, logger);
            return coordinatorInstance;
        } 
        return coordinatorInstance;
    }

    // ===========================================================================

    public float[] callPredictionsBatch(List<DataMessage> batch) {

        if (batch == null || batch.isEmpty()) {
            logger.log("batch is empty");
            return new float[]{-1f, -1f};
        }

        List<float[]> featureList = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        for (DataMessage msg : batch) {
            if (msg == null) continue;

            float[] feats = msg.features;
            if (feats == null) {
                logger.log("Null features in DataMessage");
                continue;
            }

            if (feats.length != NUM_FEATURES) {
                logger.log("Wrong features length. Got " + feats.length + " but NUM_FEATURES = " + NUM_FEATURES);
                System.out.println("Wrong features length. Got " + feats.length + " but NUM_FEATURES = " + NUM_FEATURES);
                continue; // skip non-conforming record
            }

            featureList.add(feats);
            labels.add(msg.label);
        }

        int nSamples = featureList.size();
        if (nSamples == 0) {
            logger.log("No samples after parsing batch");
            return new float[]{-1f, -1f};
        }

        float[][] data = new float[nSamples][NUM_FEATURES];
        for (int i = 0; i < nSamples; i++) {
            // copy to avoid surprises if upstream reuses arrays (optional but safe)
            System.arraycopy(featureList.get(i), 0, data[i], 0, NUM_FEATURES);
        }

        INDArray X = Nd4j.create(data);              // [batch, NUM_FEATURES]
        INDArray probs = model.output(X, false);     // [batch, NUM_CLASSES] or [batch,1] if sigmoid

        
        if (probs == null || probs.size(0) == 0) {
            logger.log("Empty probs batch");
            return new float[]{-1f, -1f};
        }

        int nCorrect = 0;
        float loss = 0f;
        float[] sampleLosses = new float[nSamples];

        // BINARY CASE (SIGMOID) ======================================================================

        if (NEURAL_OUTPUT == 1) {

            for (int i = 0; i < nSamples; i++) {

                float p = probs.getFloat(i, 0);

                if (p < 1e-7f) p = 1e-7f;
                if (p > 1f - 1e-7f) p = 1f - 1e-7f;

                int label = labels.get(i);   // 0 or 1

                int pred = (p >= 0.5f) ? 1 : 0;
                if (pred == label) nCorrect++;

                sampleLosses[i] = LossFunction.compute_loss(p, label);
            }

        // MULTI-CLASS CASE (SOFTMAX) =================================================================

        } else {

            INDArray argMax = probs.argMax(1);   // [batch]

            for (int i = 0; i < nSamples; i++) {
                int pred = argMax.getInt(i);
                int label = labels.get(i);

                if (pred == label) nCorrect++;

                float[] probabilities = probs.getRow(i).toFloatVector();

                sampleLosses[i] = LossFunction.compute_loss(probabilities, label);
            }
        }

        // Combine Losses from Multiple Samples =======================================================

        if ("TOP_K".equals(LOSS_COMBINE)) {
            loss = LossFunction.topKAverage(sampleLosses);

        } else if ("SUM".equals(LOSS_COMBINE)) {
            loss = LossFunction.sum(sampleLosses);

        } else if ("AVG".equals(LOSS_COMBINE)) {
            loss = LossFunction.average(sampleLosses);

        } else {
            loss = LossFunction.average(sampleLosses);
        }

        if (Float.isNaN(loss) || Float.isInfinite(loss)) {
            logger.log("loss is NaN/Inf, X length: " + X.length());
            return new float[]{-1f, -1f};
        }

        if (nCorrect == 0) {
            logger.log("nCorrect == 0");
        }

        float accuracy = (float) nCorrect / nSamples;
        return new float[]{accuracy, loss, nSamples, nCorrect};
    }

        
    // ===========================================================================

    public String predictSingleBest(DataMessage msg) {
        if (msg == null || msg.features == null || msg.features.length != NUM_FEATURES) {
            return null;
        }

        try {
            INDArray X = Nd4j.create(msg.features).reshape(1, NUM_FEATURES);
            INDArray probs = bestModel.output(X, false);
            int pred = probs.argMax(1).getInt(0);

            Map<String, Object> out = new HashMap<>();
            out.put("sample_index", msg.sampleIndex);
            out.put("prediction", pred);

            return MAPPER.writeValueAsString(out);

        } catch (Exception e) {
            logger.log("Error in predictSingle: " + e.getMessage());
            return null;
        }
    }
}




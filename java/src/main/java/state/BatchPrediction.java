
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
    private static final String COMBINE_LOSS = cfg.COMBINE_LOSS;
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT; 

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final MultiLayerNetwork model;
    private final MultiLayerNetwork bestModel;

    private static BatchPrediction coordinatorInstance = null;

    private final CustomLogger logger;
    private final boolean isCoordinator;

    private static boolean checked = false;

    private long start = System.nanoTime();
    private long end = System.nanoTime();

    // for Worker =======================================================================================================

    public BatchPrediction(MultiLayerNetwork model, CustomLogger logger) {
        this.model = model;
        this.bestModel = null;
        this.logger = logger;
        this.isCoordinator = false;
    }

    // for Coordinator ==================================================================================================

    public BatchPrediction(MultiLayerNetwork model, MultiLayerNetwork bestModel, CustomLogger logger) {
        this.model = model;
        this.bestModel = bestModel;
        this.logger = logger;
        this.isCoordinator = true;
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

            float[] features = msg.features;
            if (features == null) {
                logger.log("Null features in DataMessage");
                continue;
            }

            if (features.length != NUM_FEATURES) {
                logger.log("Wrong features length. Got " + features.length + " but NUM_FEATURES = " + NUM_FEATURES);
                System.out.println("Wrong features length. Got " + features.length + " but NUM_FEATURES = " + NUM_FEATURES);
                continue; // skip non-conforming record
            }

            featureList.add(features);
            labels.add(msg.label);
        }

        int nSamples = featureList.size();
        if (nSamples == 0) {
            logger.log("No samples after parsing batch");
            return new float[]{-1f, -1f};
        }

        float[][] data = new float[nSamples][NUM_FEATURES];     // matrix of samples and features
        for (int i = 0; i < nSamples; i++) {
            System.arraycopy(featureList.get(i), 0, data[i], 0, NUM_FEATURES);
        }


        // Start Forward Pass ===============================================================================
        start = System.nanoTime();

        INDArray X;
        
        if("cifar3".equals(DATASET)) {
            // data is float[nSamples][3072] (flattened NHWC from Python)
            INDArray X2d = Nd4j.create(data);                  // [batch, 3072]
            INDArray X4d = X2d.reshape(nSamples, 32, 32, 3);     // [batch, 32, 32, 3]  (NHWC)

            if(checked == false) {
                float r = X4d.getFloat(0, 0, 0, 0);
                float g = X4d.getFloat(0, 0, 0, 1);
                float b = X4d.getFloat(0, 0, 0, 2);
                System.out.println("first pixel rgb = " + r + ", " + g + ", " + b);
                checked = true;
            }

            X = X4d.permute(0, 3, 1, 2).dup();        // [batch, 3, 32, 32]

        } else if("mnist".equals(DATASET) || "mnist4".equals(DATASET) ) {

            INDArray X2d = Nd4j.create(data);          // [batch, 784]
            X = X2d.reshape(X2d.size(0), 1, 28, 28);
            // INDArray probs = model.output(X4d, false);

        } else {
            X = Nd4j.create(data);                     // [batch, NUM_FEATURES]
        }

        INDArray probs = model.output(X, false);     // [batch, NUM_CLASSES] or [batch,1] if sigmoid

        end = System.nanoTime();
        // End Forward Pass ===============================================================================

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

        if ("TOP_K".equals(COMBINE_LOSS)) {
            loss = LossFunction.topKAverage(sampleLosses);

        } else if ("SUM".equals(COMBINE_LOSS)) {
            loss = LossFunction.sum(sampleLosses);

        } else if ("AVG".equals(COMBINE_LOSS)) {
            loss = LossFunction.average(sampleLosses);

        } else {
            loss = LossFunction.average(sampleLosses);
        }

        // ===========================================================================================
        
        if (Float.isNaN(loss) || Float.isInfinite(loss)) {
            logger.log("loss is NaN/Inf, X length: " + X.length());
            return new float[]{-1f, -1f};
        }

        if (nCorrect == 0) {
            logger.log("nCorrect == 0");
        }

        float accuracy = (float) nCorrect / nSamples;
        // if(isCoordinator) {
        //     logger.log("BatchPrediction weights sample: " + Dl4jParamUtils.sampleFlat(Dl4jParamUtils.modelToFlatList(model), SAMPLING_CONSTANT)
        //         + ", accuracy: " + accuracy + ", with nSamples: " + nSamples + ", nCorrect: " + nCorrect );
        // }

        float forwardMs = (end - start) / 1_000_000f;

        return new float[]{accuracy, loss, nSamples, nCorrect, forwardMs};
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




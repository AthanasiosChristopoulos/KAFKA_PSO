
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

public class BatchPrediction {

    private static final Config cfg = Config.getInstance();
    public final int NEURAL_INPUT = cfg.NEURAL_INPUT;
    public final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;
    public final String DATASET = cfg.DATASET;
 
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final MultiLayerNetwork model;
    private final Stats stats;
    private static BatchPrediction coordinatorInstance = null;

    private final CustomLogger logger;

    public BatchPrediction(MultiLayerNetwork model, Stats stats, CustomLogger logger) {
        this.model = model;
        this.stats = stats;
        this.logger = logger;
    }

    public static BatchPrediction getCoordinatorInstance(MultiLayerNetwork model, Stats stats, CustomLogger logger) {
        if(coordinatorInstance == null) {
            coordinatorInstance = new BatchPrediction(model, stats, logger);
            return coordinatorInstance;
        } 
        return coordinatorInstance;
    }

    // ===========================================================================

    public float[] callPredictionsBatch(List<String> jsonValues) {

        if (jsonValues == null || jsonValues.isEmpty()) {
            logger.log("json values are empty");
            return new float[]{-1f, -1f};
        }

        List<float[]> featureList = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        for (String value : jsonValues) {   // extract features, labels out of json values for data batch
            try {
                Map<String, Object> obj = MAPPER.readValue(value, new TypeReference<Map<String, Object>>() {});

                List<?> featList = (List<?>) obj.get("features");
                if (featList == null) {
                    logger.log("Missing 'features' key in JSON");
                    continue;
                }

                if (featList.size() != NEURAL_INPUT) {
                    logger.log("Wrong features length. Got " + featList.size() + " but NEURAL_INPUT = " + NEURAL_INPUT);
                    System.out.println("Wrong features length. Waiting for " + featList.size() + ", but NEURAL_INPUT = " + NEURAL_INPUT);
                    continue;   // skip non-conforming record
                }

                float[] features = new float[NEURAL_INPUT];
                for (int i = 0; i < NEURAL_INPUT; i++) {
                    features[i] = ((Number) featList.get(i)).floatValue();
                }

                if (!obj.containsKey("label")) {
                    logger.log("Missing 'label' key in JSON");
                    continue;
                }

                int label = ((Number) obj.get("label")).intValue();   // 0/1 or class index

                featureList.add(features);
                labels.add(label);

            } catch (Exception e) {
                logger.log("Error parsing JSON: " + e.getMessage());
            }
        }

        int nSamples = featureList.size();
        if (nSamples == 0) {
            logger.log("No samples after parsing jsonValues");
            return new float[]{-1f, -1f};
        }

        float[][] data = new float[nSamples][NEURAL_INPUT];
        for (int i = 0; i < nSamples; i++) {
            data[i] = featureList.get(i);
        }

        INDArray X = Nd4j.create(data);              // [batch, NEURAL_INPUT]
        INDArray probs = model.output(X, false);     // [batch, NEURAL_OUTPUT] or [batch,1] if sigmoid

        if (probs == null || probs.size(0) == 0) {
            logger.log("Empty probs batch");
            return new float[]{-1f, -1f};
        }

        int nCorrect = 0;
        float loss = 0f;

        // ====== SIGMOID / BINARY CASE ======

        if ("bank".equals(this.DATASET) || "adult".equals(this.DATASET)) {
            logger.log("I am here_1");
            for (int i = 0; i < nSamples; i++) {

                float p = probs.getFloat(i, 0);

                // Clamp to avoid log(0)
                if (p < 1e-7f) p = 1e-7f;
                if (p > 1f - 1e-7f) p = 1f - 1e-7f;

                int label = labels.get(i);   // 0 or 1

                int pred = (p >= 0.5f) ? 1 : 0;  // Prediction with threshold 0.5
                if (pred == label) {
                    nCorrect++;
                }

                float sampleLoss = LossFunction.compute_loss_sigmoid(p, label);
                loss += sampleLoss;
            }

        // ====== SOFTMAX / MULTI-CLASS CASE ======

        } else {
            INDArray argMax = probs.argMax(1);   // [batch]

            for (int i = 0; i < nSamples; i++) {
                int pred = argMax.getInt(i);     // index of class with max prob
                int label = labels.get(i);

                if (pred == label) {
                    nCorrect++;
                }

                float[] probabilities = probs.getRow(i).toFloatVector();
                loss += LossFunction.compute_loss(probabilities, label);
            }
        }

        if (Float.isNaN(loss) || Float.isInfinite(loss)) {
            logger.log("loss is NaN/Inf, X length: " + X.length());
            return new float[]{-1f, -1f};
        }

        if (nCorrect == 0) {
            logger.log("nCorrect == 0");
        }

        float accuracy = (float) nCorrect / nSamples;
        // float avgLoss = loss / nSamples;         // not averaging loss seems to lead to higher performance
        // return new float[]{accuracy, avgLoss};
        logger.log("I am here_2 " + "nSamples: " + nSamples );

        return new float[]{accuracy, loss, nSamples, nCorrect};
    }

    
    // ===========================================================================

    public String predictSingle(String jsonValue) {
        if (jsonValue == null || jsonValue.isEmpty()) {
            return null;
        }

        try {
            Map<String, Object> obj = MAPPER.readValue(
                    jsonValue, new TypeReference<Map<String, Object>>() {}
            );

            List<?> featList = (List<?>) obj.get("features");
            if (featList == null || featList.size() != NEURAL_INPUT) {
                return null; // bad record
            }

            float[] features = new float[NEURAL_INPUT];
            for (int i = 0; i < NEURAL_INPUT; i++) {
                features[i] = ((Number) featList.get(i)).floatValue();
            }

            // Create [1, NEURAL_INPUT] INDArray
            INDArray X = Nd4j.create(features).reshape(1, NEURAL_INPUT);
            INDArray probs = model.output(X, false);   // [1, NEURAL_OUTPUT]
            int pred = probs.argMax(1).getInt(0);      // single prediction

            Object sampleIndex = obj.get("sample_index");
            Object label_name = obj.get("label_name");

            Map<String, Object> out = new HashMap<>();
            out.put("sample_index", sampleIndex);
            out.put("prediction", pred);
            out.put("true_label_name", label_name);

            return MAPPER.writeValueAsString(out);
        } catch (Exception e) {
            logger.log("Error in predictSingle: " + e.getMessage());
            return null;
        }
    }

}




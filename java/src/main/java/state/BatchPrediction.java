
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

import utils.*;

public class BatchPrediction {

    private static final Config cfg = Config.getInstance();
    public final int NEURAL_INPUT = cfg.NEURAL_INPUT;
    public final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final MultiLayerNetwork model;
    private final Stats stats;
    private static BatchPrediction coordinatorInstance = null;

    public BatchPrediction(MultiLayerNetwork model, Stats stats) {
        this.model = model;
        this.stats = stats;
    }

    public static BatchPrediction getCoordinatorInstance(MultiLayerNetwork model, Stats stats) {
        if(coordinatorInstance == null) {
            coordinatorInstance = new BatchPrediction(model, stats);
            return coordinatorInstance;
        } 
        return coordinatorInstance;
    }

    // ===========================================================================

    public void callPredictionsBatch(List<String> jsonValues) {
        if (jsonValues == null || jsonValues.isEmpty()) {
            return;
        }

        List<float[]> featureList = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        // Parse JSONs into features + labels
        for (String value : jsonValues) {
            try {
                Map<String, Object> obj = MAPPER.readValue(value, new TypeReference<Map<String, Object>>() {});

                List<?> featList = (List<?>) obj.get("features");
                if (featList == null || featList.size() != NEURAL_INPUT) {
                    continue;       // skip non conforming record
                }
                float[] features = new float[NEURAL_INPUT];
                for (int i = 0; i < NEURAL_INPUT; i++) {
                    features[i] = ((Number) featList.get(i)).floatValue();
                }
                int label = ((Number) obj.get("label")).intValue();

                featureList.add(features);
                labels.add(label);

            } catch (Exception e) {
                System.out.println("Error");
            }
        }

        int nSamples = featureList.size();
        if (nSamples == 0) {
            return;
        }

        float[][] data = new float[nSamples][NEURAL_INPUT];
        for (int i = 0; i < nSamples; i++) {
            data[i] = featureList.get(i);
        }
        
        INDArray X = Nd4j.create(data);              // [batch, NEURAL_INPUT]
        INDArray probs = model.output(X, false);     // [batch, NEURAL_OUTPUT]
        INDArray argMax = probs.argMax(1);           // [batch]

        int nCorrect = 0;
        float loss = 0;
        
        for (int i = 0; i < nSamples; i++) {
            
            // Measure Accuracy
            int pred = argMax.getInt(i);
            int label = labels.get(i);
            if (pred == label) {
                nCorrect++;
            }

            // Measure Loss
            float[] probabilities = probs.getRow(i).toFloatVector();      // get the probabilities for current sample 
            loss += LossFunction.compute_loss(probabilities, label);
        }
   
        stats.addBatch(nSamples, nCorrect, loss); // Update this worker's stats
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
            System.out.println("Error in predictSingle: " + e.getMessage());
            return null;
        }
    }

}

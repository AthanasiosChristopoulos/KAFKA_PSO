
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

        List<double[]> featureList = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();

        // Parse JSONs into features + labels
        for (String value : jsonValues) {
            try {
                Map<String, Object> obj = MAPPER.readValue(value, new TypeReference<Map<String, Object>>() {});

                List<?> featList = (List<?>) obj.get("features");
                if (featList == null || featList.size() != 4) {
                    continue;       // skip non conforming record
                }
                double[] features = new double[4];
                for (int i = 0; i < 4; i++) {
                    features[i] = ((Number) featList.get(i)).doubleValue();
                }
                int label = ((Number) obj.get("label")).intValue();

                featureList.add(features);
                labels.add(label);
            } catch (Exception e) {
                System.out.println("Error");
            }
        }

        int nPredictions = featureList.size();
        if (nPredictions == 0) {
            return;
        }

        double[][] data = new double[nPredictions][4];
        for (int i = 0; i < nPredictions; i++) {
            data[i] = featureList.get(i);
        }
        
        INDArray X = Nd4j.create(data);              // [batch, 4]
        INDArray probs = model.output(X, false);     // [batch, 3]
        INDArray argMax = probs.argMax(1);           // [batch]

        int nCorrect = 0;
        double loss = 0;
        
        for (int i = 0; i < nPredictions; i++) {
            
            // Measure Accuracy
            int pred = argMax.getInt(i);
            int label = labels.get(i);
            if (pred == label) {
                nCorrect++;
            }

            // Measure Loss
            double[] probabilities = probs.getRow(i).toDoubleVector();
            loss += LossFunction.compute(probabilities, label);
        }
   
        stats.addBatch(nPredictions, nCorrect, loss); // Update this worker's stats
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
            if (featList == null || featList.size() != 4) {
                return null; // bad record
            }

            double[] features = new double[4];
            for (int i = 0; i < 4; i++) {
                features[i] = ((Number) featList.get(i)).doubleValue();
            }

            // Create [1,4] INDArray
            INDArray X = Nd4j.create(features).reshape(1, 4);
            INDArray probs = model.output(X, false);   // [1, 3]
            int pred = probs.argMax(1).getInt(0);      // single prediction

            Object sampleIndex = obj.get("sample_index");
            Map<String, Object> out = new HashMap<>();
            out.put("sample_index", sampleIndex);
            out.put("prediction", pred);

            return MAPPER.writeValueAsString(out);
        } catch (Exception e) {
            System.out.println("Error in predictSingle: " + e.getMessage());
            return null;
        }
    }

}

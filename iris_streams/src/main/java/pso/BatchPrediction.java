
package pso;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class BatchPrediction {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final MultiLayerNetwork model;
    private final WorkerStats stats;

    public BatchPrediction(MultiLayerNetwork model, WorkerStats stats) {
        this.model = model;
        this.stats = stats;
    }

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
                    continue; // skip bad record
                }
                double[] features = new double[4];
                for (int i = 0; i < 4; i++) {
                    features[i] = ((Number) featList.get(i)).doubleValue();
                }
                int label = ((Number) obj.get("label")).intValue();

                featureList.add(features);
                labels.add(label);
            } catch (Exception e) {
                // skip malformed record
            }
        }

        int batchSize = featureList.size();
        if (batchSize == 0) {
            return;
        }

        double[][] data = new double[batchSize][4];
        for (int i = 0; i < batchSize; i++) {
            data[i] = featureList.get(i);
        }
        
        INDArray X = Nd4j.create(data);              // [batch, 4]
        INDArray probs = model.output(X, false);     // [batch, 3]
        INDArray argMax = probs.argMax(1);           // [batch]

        int correctBatch = 0;
        for (int i = 0; i < batchSize; i++) {
            int pred = argMax.getInt(i);
            int label = labels.get(i);
            if (pred == label) {
                correctBatch++;
            }
        }

        // Update this worker's stats
        stats.addBatch(batchSize, correctBatch);
    }
}

package evaluate;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import java.io.IOException;
import pso.Dl4jParamUtils;

public class EvaluateIrisModel {

    public static void main(String[] args) throws IOException {
        System.out.println("Executing model Evaluation ...");

        File modelFile = new File("models/iris-global-model.zip");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);

        System.out.println(Arrays.toString(Dl4jParamUtils.modelToFlatList(model)));

        // 2. Create an iterator over the Iris dataset
        int batchSize = 150; // full dataset
        int numClasses = 3;
        DataSetIterator iter = new IrisDataSetIterator(batchSize, batchSize);
        
        // ===== Print the first 10 samples from the dataset =====
        var ds = iter.next();  // entire dataset (150 rows)

        System.out.println("=== Showing first 10 rows of Iris dataset used by DL4J ===");
        for (int i = 0; i < 10; i++) {
            double[] features = ds.getFeatures().getRow(i).toDoubleVector();  // [4]
            double[] labels = ds.getLabels().getRow(i).toDoubleVector();      // one-hot [3]

            System.out.printf(
                "Row %d: features=%s  label=%s%n",
                i,
                Arrays.toString(features),
                Arrays.toString(labels)
            );
        }
        System.out.println("===========================================================");
        // 3. Evaluate
        Evaluation eval = model.evaluate(iter);

        // 4. Print accuracy
        double accuracy = eval.accuracy();
        System.out.println("Iris model accuracy: " + accuracy);
    }
}

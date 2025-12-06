package evaluate;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import utils.*;

public class EvaluateIrisModel {

    public static void main(String[] args) throws IOException {
        System.out.println("Executing model Evaluation ...");

        Config cfg = Config.getInstance();
        String SAVE_MODEL_NAME = cfg.SAVE_MODEL_NAME;

        File modelFile = new File("models/" + SAVE_MODEL_NAME + "-dl4j.zip");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);

        System.out.println(Arrays.toString(Dl4jParamUtils.modelToFlatList(model)));

        int batchSize = 150; 
        int numClasses = 3;
        IrisDataSetIterator iter = new IrisDataSetIterator(batchSize, batchSize);
        var ds = iter.next(); 

        INDArray X = ds.getFeatures();  // shape [150, 4]
        INDArray y = ds.getLabels();    // shape [150, 3]

        INDArray mean = X.mean(0);  // shape [1, 4]
        INDArray std = X.std(0);    // shape [1, 4]

        std = std.add(1e-8);

        INDArray X_scaled = X.subRowVector(mean).divRowVector(std);

        DataSet scaledDataSet = new DataSet(X_scaled, y);

        // ===== Print first 10 scaled samples =====
        System.out.println("=== First 10 SCALED Iris samples (DL4J) ===");
        for (int i = 0; i < 10; i++) {
            double[] features = X_scaled.getRow(i).toDoubleVector();
            double[] labels = y.getRow(i).toDoubleVector();

            System.out.printf(
                "Row %d: features=%s  label=%s%n",
                i,
                Arrays.toString(features),
                Arrays.toString(labels)
            );
        }
        System.out.println("==========================================");


        // 5. Evaluate model using scaled data
        Evaluation eval = new Evaluation(numClasses);
        INDArray output = model.output(X_scaled, false);
        eval.eval(y, output);

        double accuracy = eval.accuracy();
        System.out.println("Iris model accuracy (scaled): " + accuracy);

    }
}

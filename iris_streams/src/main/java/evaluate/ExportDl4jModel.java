package evaluate;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class ExportDl4jModel {

    public static void main(String[] args) throws IOException {
        System.out.println("Exporting Dl4J Model");
        // 1. Load the trained DL4J model from your saved zip
        File modelFile = new File("models/iris-global-model.zip");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);

        // 2. Build a JSON structure: list of layers, each with in_size, out_size, weights, bias
        List<Map<String, Object>> layers = new ArrayList<>();

        for (int layerIdx = 0; layerIdx < model.getnLayers(); layerIdx++) {
            Layer l = model.getLayer(layerIdx);
            Map<String, INDArray> params = l.paramTable();

            INDArray W = params.get("W");
            INDArray b = params.get("b");

            if (W == null || b == null) {
                // skip layers without W/b (e.g., some special layers)
                continue;
            }

            long inSize = W.size(0);   // DL4J: [in, out]
            long outSize = W.size(1);

            // We’ll export weights as shape [out_size][in_size],
            // i.e. each row = one neuron (this matches PyTorch Linear weight shape).
            List<List<Double>> weightRows = new ArrayList<>();
            for (int j = 0; j < outSize; j++) {            // neuron index (output)
                List<Double> row = new ArrayList<>();
                for (int i = 0; i < inSize; i++) {        // input index
                    double w = W.getDouble(i, j);         // DL4J stores (in, out)
                    row.add(w);
                }
                weightRows.add(row);
            }

            // Bias: length = outSize
            List<Double> biasList = new ArrayList<>();
            for (int j = 0; j < outSize; j++) {
                biasList.add(b.getDouble(j));
            }

            Map<String, Object> layerMap = new HashMap<>();
            layerMap.put("layer_index", layerIdx);
            layerMap.put("type", "dense");
            layerMap.put("in_size", inSize);
            layerMap.put("out_size", outSize);
            layerMap.put("weights", weightRows);
            layerMap.put("bias", biasList);

            layers.add(layerMap);
        }

        // 3. Wrap into a top-level object
        Map<String, Object> root = new HashMap<>();
        root.put("layers", layers);

        // OPTIONAL: store some metadata (architecture) if you want
        root.put("input_dim", 4);
        root.put("num_classes", 3);

        // 4. Write JSON to file
        File outFile = new File("models/iris-dl4j-export.json");
        ObjectMapper mapper = new ObjectMapper();
        mapper.writerWithDefaultPrettyPrinter().writeValue(outFile, root);

        System.out.println("Exported DL4J model to: " + outFile.getAbsolutePath());
    }
}

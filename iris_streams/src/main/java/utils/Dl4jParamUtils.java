package utils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.io.BufferedWriter;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Collection;
import java.util.*;


public class Dl4jParamUtils {

    public static double[] modelToFlatList(MultiLayerNetwork model) {
        List<Double> flatList = new ArrayList<>();

        for (int layerIdx = 0; layerIdx < model.getnLayers(); layerIdx++) {
            Layer l = model.getLayer(layerIdx);
            Map<String, INDArray> params = l.paramTable();

            INDArray W = params.get("W");
            INDArray b = params.get("b");

            if (W == null || b == null) {
                continue;
            }

            long inSize = W.size(0);
            long outSize = W.size(1);

            // same convention: j = neuron index, i = input index
            for (int j = 0; j < outSize; j++) {
                for (int i = 0; i < inSize; i++) {
                    flatList.add(W.getDouble(i, j));
                }
                // bias for neuron j
                flatList.add(b.getDouble(j));
            }
        }

        double[] flat = new double[flatList.size()];
        for (int i = 0; i < flat.length; i++) {
            flat[i] = flatList.get(i);
        }
        return flat;
    }
    
    //=====================================================================================================

    public static void updateModel(MultiLayerNetwork model, double[] flat) {
        int idx = 0;

        for (int layerIdx = 0; layerIdx < model.getnLayers(); layerIdx++) {
            Layer l = model.getLayer(layerIdx);
            Map<String, INDArray> params = l.paramTable();

            INDArray W = params.get("W");
            INDArray b = params.get("b");

            if (W == null || b == null) {
                continue;
            }

            long inSize = W.size(0);
            long outSize = W.size(1);

            for (int j = 0; j < outSize; j++) {
                for (int i = 0; i < inSize; i++) {
                    W.putScalar(i, j, flat[idx++]);
                }
                b.putScalar(j, flat[idx++]);
            }

            // Push updated arrays back into the layer
            l.setParam("W", W);
            l.setParam("b", b);
        }

        if (idx != flat.length) {
            throw new IllegalArgumentException(
                    "Flat vector length mismatch, consumed " + idx + " of " + flat.length
            );
        }
    }

    //=====================================================================================================

    public static String sampleFlat(double[] flat) {
        int n = Math.min(3, flat.length);

        StringBuilder sb = new StringBuilder();
        sb.append("[");

        for (int i = 0; i < n; i++) {
            sb.append(String.format("%.5f", flat[i]));
            if (i < n - 1) sb.append(", ");
        }

        if (flat.length > 3) {
            sb.append(", ...");
        }

        sb.append("]");

        return sb.toString();
    }

    //=====================================================================================================

    public static String sampleFlats(Collection<double[]> flats) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");

        int idx = 0;
        int size = flats.size();

        for (double[] flat : flats) {
            sb.append(sampleFlat(flat));   // reuse your existing sampling function

            if (idx < size - 1) {
                sb.append(",\n");
            }
            idx++;
        }

        sb.append("]");

        return sb.toString();
    }

    //=====================================================================================================

    public static void saveModel(MultiLayerNetwork model, String modelFileName) {

        File dir = new File("models"); // create Models Directory if it doesnt exist
        if (!dir.exists()) {
            dir.mkdirs();
        }

        // ===================== Save the model using DL4J =====================
        try {
            File modelFile = new File(dir, modelFileName + "-dl4j.zip");
            ModelSerializer.writeModel(model, modelFile, true);

            System.out.println("Saved global model to: " + modelFile.getAbsolutePath());

        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Failed to save model: " + e.getMessage());
        }

        // ===================== Save the model as a list =====================

        double[] flat = modelToFlatList(model);
        String filenameFlat = "models/" + modelFileName + "-flat.txt";

        try {
            try (BufferedWriter writer = Files.newBufferedWriter(
                    Paths.get(filenameFlat),
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                for (double weight : flat) {
                    writer.write(Double.toString(weight));
                    writer.newLine();
                }
            }

            System.out.println("Saved flat weights to" + filenameFlat + ", with length=" + flat.length);

        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Failed to save flat weights: " + e.getMessage());
        }

    }

}

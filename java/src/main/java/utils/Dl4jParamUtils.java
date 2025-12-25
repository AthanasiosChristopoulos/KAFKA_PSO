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


public class Dl4jParamUtils {   

    private static Config cfg = Config.getInstance();
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;

    public static float[] modelToFlatList(MultiLayerNetwork model) {

        List<Float> flatList = new ArrayList<>();

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

            // convention W_L(i, j): j = neuron index, i = input index (input weights), L = number of layer
            for (int j = 0; j < outSize; j++) {

                for (int i = 0; i < inSize; i++) {  // All inputs to neuron j
                    flatList.add(W.getFloat(i, j));
                }

                flatList.add(b.getFloat(j));       // Bias for neuron j
            }
        }

        float[] flat = new float[flatList.size()];
        for (int i = 0; i < flat.length; i++) {
            flat[i] = flatList.get(i);
        }
        return flat;
    }
    
    //=====================================================================================================

    public static void updateModel(MultiLayerNetwork model, float[] flat) {
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

    public static String sampleFlat(float[] flat, int numberOfSamples) {

        StringBuilder sb = new StringBuilder();
        sb.append("[");

        for (int i = 0; i < numberOfSamples; i++) {
            sb.append(String.format("%.5f", flat[i]));
            if (i < numberOfSamples - 1) sb.append(", ");
        }

        sb.append(", ...]");
        return sb.toString();
    }

    //=====================================================================================================

    public static String sampleFlatSorted(float[] flat, int numberOfSamples) {

        int[] topIdx = new int[numberOfSamples];
        for (int i = 0; i < numberOfSamples; i++) {
            topIdx[i] = -1;
        }
        
        for (int i = 0; i < flat.length; i++) {
            float amp = Math.abs(flat[i]);

            for (int pos = 0; pos < numberOfSamples; pos++) {
                if (topIdx[pos] == -1 || amp > Math.abs(flat[topIdx[pos]])) {

                    for (int k = numberOfSamples - 1; k > pos; k--) {
                        topIdx[k] = topIdx[k - 1];
                    }

                    topIdx[pos] = i;
                    break;
                }
            }
        }

        StringBuilder sb = new StringBuilder();
        sb.append("[");

        for (int i = 0; i < numberOfSamples; i++) {
            sb.append(String.format("%.5f", flat[topIdx[i]]));
            if (i < numberOfSamples - 1) {
                sb.append(", ");
            }
        }

        sb.append(", ...]");
        return sb.toString();
    }

    //=====================================================================================================

    public static String sampleFlats(Collection<float[]> flats) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");

        int idx = 0;
        int size = flats.size();

        for (float[] flat : flats) {
            sb.append(sampleFlat(flat, SAMPLING_CONSTANT));   

            if (idx < size - 1) {
                sb.append(",\n");
            }
            idx++;
        }

        sb.append("]");

        return sb.toString();
    }

    //=====================================================================================================

    public static float magnitude(float[] flat) {
        float sum = 0f;

        for (float v : flat) {
            sum += v * v; 
        }

        return (float) Math.sqrt(sum);
    }

    //=====================================================================================================

    public static void saveModel(MultiLayerNetwork model) {

        Config cfg = Config.getInstance();
        String SAVE_MODEL_NAME = cfg.SAVE_MODEL_NAME;

        File dir = new File("models"); // create Models Directory if it doesnt exist
        if (!dir.exists()) {
            dir.mkdirs();
        }

        // ===================== Save the model using DL4J =====================
        try {
            File modelFile = new File(dir, SAVE_MODEL_NAME + "-dl4j.zip");
            ModelSerializer.writeModel(model, modelFile, true);

            System.out.println("Saved global model to: " + modelFile.getAbsolutePath());

        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Failed to save model: " + e.getMessage());
        }

        // ===================== Save the model as a list =====================

        float[] flat = modelToFlatList(model);
        String filenameFlat = "models/" + SAVE_MODEL_NAME + "-flat.txt";

        try {
            try (BufferedWriter writer = Files.newBufferedWriter(
                    Paths.get(filenameFlat),
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                for (float weight : flat) {
                    writer.write(Float.toString(weight));
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

package dl4j_models;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import utils.Config;

import java.io.File;
import java.io.IOException;
import java.io.BufferedWriter;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

import java.util.Collection;
import java.util.Arrays;

public class Dl4jParamUtils {   

    private static Config cfg = Config.getInstance();
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;
    public static final float MONITORING_THRESHOLD = cfg.MONITORING_THRESHOLD;
	public static int POINTS_PER_AXIS = cfg.POINTS_PER_AXIS;
    public static String SAVE_MODEL_NAME = cfg.SAVE_MODEL_NAME;

    //=====================================================================================================
    // model.params() => both weights and biases (actually all other trainable parameters)

    public static float[] modelToFlatList(PsoModel model) {    // Serializa model into float[]
        return model.params().toFloatVector();      // model.params() returns one flat vector that contains every parameter in the model. specific order chosen by DL4J

    }   // saves parameters in this order:
            // For CNNs (Convolutional Layer): [biases, parameters]
            // For FNNs (Dense Layer): [weights, biases]: Usually its: [Layer0_weights, Layer0_biases, Layer1_weights, Layer1_biases, ... ]

    //=====================================================================================================

    public static float[] modelToFlatHead(PsoModel model, int start) {
        float[] full = model.params().toFloatVector();
        return Arrays.copyOfRange(full, start, full.length);
    }

    //=====================================================================================================
    //=====================================================================================================

    public static void updateModel(PsoModel model, float[] flat) {
        INDArray params = model.params();   // a pointer to the actual parameter buffer owned by that model
        params.data().setData(flat);  
    }
    
    //=====================================================================================================

    public static void updateModelHead(PsoModel model, float[] headFlat, int start) {

        INDArray p = model.params(); 
        int end = (int) model.numParams();

        if (headFlat.length != (end - start)) {
            throw new IllegalArgumentException("Head length mismatch. expected = " + (end - start) 
                + " got = " + headFlat.length + ", with start = " + start + ", and end = " + end);
        }

        INDArray headView = p.get(NDArrayIndex.interval(start, end));
        headView.data().setData(headFlat); 
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

    public static float rms(float[] flat) {    // this is RMS for vectors
        float sum = 0f;

        for (float v : flat) {
            sum += v * v; 
        }

        return (float) Math.sqrt(sum / flat.length);
    }

    //=====================================================================================================

    public static float rmsScaled(float[] flat, int scale) {
        float sum = 0f;

        for (float v : flat) {
            sum += v * v;
        }

        float rms = (float) Math.sqrt(sum / flat.length) * scale;
        return round(rms, 3);
    }

    //=====================================================================================================

    public static float round(float number, int decimals) {

        float scale_factor = (float) Math.pow(10, decimals);
        return Math.round(number * scale_factor) / scale_factor;
    }

    //=====================================================================================================

    public static void saveModel(PsoModel model, String name, int idx) {

        File dir = new File("models"); // create Models Directory if it doesnt exist
        if (!dir.exists()) {
            dir.mkdirs();
        }

        // Save the model as a list =====================
        float[] flat;

        if(cfg.USING_PRETRAINED_MODEL) {
            flat = modelToFlatHead(model, idx);
        } else {
            flat = modelToFlatList(model);
        }

        String filenameFlat = "models/" + name + "-flat.txt";

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

            System.out.println("Saved flat weights to " + filenameFlat + ", with length=" + flat.length);

        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Failed to save flat weights: " + e.getMessage());
        }
    }
}

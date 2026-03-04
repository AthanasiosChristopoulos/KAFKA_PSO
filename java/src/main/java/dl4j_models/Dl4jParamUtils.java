package dl4j_models;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.api.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.lang.reflect.Field;
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
                                                    // INDArray.toFloatVector() allocates a fresh float[] copy every call.
        return model.params().toFloatVector();      // model.params() returns one flat vector that contains every parameter in the model
                                                    // specific order chosen by DL4J
    }   // saves parameters in this order:
            // For CNNs (Convolutional Layer): [biases, parameters]
            // For FNNs (Dense Layer): [weights, biases]: Usually its: [Layer0_weights, Layer0_biases, Layer1_weights, Layer1_biases, ... ]

    //=====================================================================================================

    public static float[] modelToFlatHead(PsoModel model, int start) {
        // if(!cfg.FREEZE) {
        //     return modelToFlatList(model);
        // }
        float[] full = model.params().toFloatVector();
        return Arrays.copyOfRange(full, start, full.length);
    }

    //=====================================================================================================
    //=====================================================================================================

    public static void updateModel(PsoModel model, float[] flat) {
        INDArray params = model.params();   // a pointer to the actual parameter buffer owned by that model
        params.data().setData(flat);   // the model object doesn’t change identity, but its internal weights do.
    }
    
    //=====================================================================================================

    public static void updateModelHead(PsoModel model, float[] headFlat, int start) {
        // if(!cfg.FREEZE) {
        //     updateModel(model, headFlat);
        // }

        INDArray p = model.params(); // 1D view of the whole parameter buffer
        // int start = ParamSlices.headFlatIndex(model, headStartLayerIdx);
        int end = (int) model.numParams();

        if (headFlat.length != (end - start)) {
            throw new IllegalArgumentException("Head length mismatch. expected = " + (end - start) 
                + " got = " + headFlat.length + ", with start = " + start + ", and end = " + end);
        }

        // assign only the head range
        INDArray headView = p.get(NDArrayIndex.interval(start, end));
        // headView.assign(Nd4j.createFromArray(headFlat));
        headView.data().setData(headFlat); 
    }

    // public static void updateModel(PsoModel model, float[] flat) {
    //     model.setParams(Nd4j.createFromArray(flat)); 
    // }
    // public static void updateModel(PsoModel model, float[] flat) { // Deserialize model, from a float[] to a PsoModel model object
    //     if (flat.length != model.numParams()) {
    //         throw new IllegalArgumentException(
    //             "Expected " + model.numParams() + " params but got " + flat.length
    //         );
    //     }
    //     model.setParams(Nd4j.createFromArray(flat));    // model.setParams(flat) expects a vector in that exact same order as set by DL4J in the start
    //                                                     // updateModel(...) mutates the existing PsoModel object in place.
    // }                                                   // DL4J provides the serialization convention
    // createFromArray => then every update step creates a new GPU buffer for params.
    //=====================================================================================================
    // Decode / Encode Model Number 2:

    // public static float[] modelToFlatList(PsoModel model) {

    //     List<Float> flatList = new ArrayList<>();

    //     for (int layerIdx = 0; layerIdx < model.getnLayers(); layerIdx++) {
    //         Layer l = model.getLayer(layerIdx);
    //         Map<String, INDArray> params = l.paramTable();

    //         INDArray W = params.get("W");
    //         INDArray b = params.get("b");

    //         if (W == null || b == null) {
    //             continue;
    //         }

    //         long inSize = W.size(0);
    //         long outSize = W.size(1);

    //         // convention W_L(i, j): j = neuron index, i = input index (input weights), L = number of layer
    //         for (int j = 0; j < outSize; j++) {

    //             for (int i = 0; i < inSize; i++) {  // All inputs to neuron j
    //                 flatList.add(W.getFloat(i, j));
    //             }

    //             flatList.add(b.getFloat(j));       // Bias for neuron j
    //         }
    //     }

    //     float[] flat = new float[flatList.size()];
    //     for (int i = 0; i < flat.length; i++) {
    //         flat[i] = flatList.get(i);
    //     }
    //     return flat;
    // }
    
    // public static void updateModel(PsoModel model, float[] flat) {
    //     int idx = 0;

    //     for (int layerIdx = 0; layerIdx < model.getnLayers(); layerIdx++) {

    //         Layer l = model.getLayer(layerIdx);
    //         Map<String, INDArray> params = l.paramTable();

    //         INDArray W = params.get("W");
    //         INDArray b = params.get("b");

    //         if (W == null || b == null) {
    //             continue;
    //         }

    //         long inSize = W.size(0);
    //         long outSize = W.size(1);

    //         for (int j = 0; j < outSize; j++) {
    //             for (int i = 0; i < inSize; i++) {
    //                 W.putScalar(i, j, flat[idx++]);
    //             }
    //             b.putScalar(j, flat[idx++]);
    //         }

    //         // Push updated arrays back into the layer
    //         l.setParam("W", W);
    //         l.setParam("b", b);
    //     }

    //     if (idx != flat.length) {
    //         throw new IllegalArgumentException(
    //                 "Flat vector length mismatch, consumed " + idx + " of " + flat.length
    //         );
    //     }
    // }

    //==============================================================================================

    // public static void updateModel(PsoModel model, float[] flat) {
    //     if (flat.length != model.numParams()) {
    //         throw new IllegalArgumentException(
    //             "Expected " + model.numParams() + " params but got " + flat.length
    //         );
    //     }
    //     model.setParams(Nd4j.create(flat));
    // }

    //==============================================================================================


    // public static void updateModel(PsoModel model, float[] flat) {

    //     long expected = model.numParams();
    //     if (flat.length != expected) {
    //         throw new IllegalArgumentException(
    //                 "Flat vector length mismatch. Got " + flat.length + " but model.numParams() = " + expected
    //         );
    //     }

    //     // Get the model's 1D parameter vector view
    //     INDArray p = model.params(); // rank-1 view

    //     // Fill it (1D indexing is always valid)
    //     for (int i = 0; i < flat.length; i++) {
    //         p.putScalar(i, flat[i]);  // <-- NOTE: 1D putScalar(index, value)
    //     }

    //     // Push back into the model
    //     model.setParams(p);
    // }

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

    // private static boolean checkIfIntersects(float[] center, double radius) {

    //     int dimensions = center.length; 
    //     int p = POINTS_PER_AXIS; 
    //     int skipped_points = 0;
    //     float[] coords = new float[dimensions];
        
    //     double tmpMin = Double.POSITIVE_INFINITY;
    //     double tmpMax = Double.NEGATIVE_INFINITY;

    //     float[] tmpMaxVector = new float[center.length];
    //     float[] tmpMinVector = new float[center.length];
    //     float[] point;
    //     float[] point_scaled_back;
    //     double val;

    //     int totalPoints = (int) Math.pow(p, dimensions);
    //     // logWriter.println("Ball center = " + center + ", radius = " + radius); 

    //     for (int i = 0; i < totalPoints; i++) {
    //         int Z = i;
    //         double squaredDist = 0.0;

    //         for (int j = 0; j < dimensions; j++) {
    //             coords[j] = (center[j] - radius + 2.0 * radius * (Z % p) / (p - 1));
    //             Z /= p;
    //             squaredDist += Math.pow(coords[j] - center[j], 2); // Eyclidean distance
    //         }

    //         // Skip points outside the sphere (turn the square grid into an actual sphere)
    //         if (squaredDist > radius * radius) {
    //             skipped_points++;
    //             continue;
    //         }
            
    //         point = new Vector(coords);

    //         val = Simulation.f(point); 

    //         if (val > tmpMax) {
    //         	tmpMaxVector = new Vector(point);
    //         	tmpMax = val;
    //         }
            
    //         if (val < tmpMin) {
    //         	tmpMinVector = new Vector(point);
    //         	tmpMin = val;
    //         }
    //     }

    //     // logWriter.println("tmpMax = " + tmpMax + ", tmpMaxVector = " + tmpMaxVector);
    //     // logWriter.println("tmpMin = " + tmpMin + ", tmpMinVector = " + tmpMinVector);

    //     boolean intersectsWithA = false;

    //     if((tmpMax > MONITORING_THRESHOLD && tmpMin < MONITORING_THRESHOLD)) {
    //         intersectsWithA = true;
    //     }
        
    //     // logWriter.println("Sphere approximation: max = " + tmpMax + ", min = " + tmpMin + ", total_points: " + totalPoints + 
    //     //     ", skipped_points: " + skipped_points + ", intersectsWithA = " + intersectsWithA);

    //     return intersectsWithA;
    // }


    //=====================================================================================================

    public static void saveModel(PsoModel model, String name, int idx) {

        File dir = new File("models"); // create Models Directory if it doesnt exist
        if (!dir.exists()) {
            dir.mkdirs();
        }

        // Save the model using DL4J =====================
        // try {
        //     File modelFile = new File(dir, name + "-dl4j.zip");
        //     ModelSerializer.writeModel(model, modelFile, true);

        //     System.out.println("Saved global model to: " + modelFile.getAbsolutePath());

        // } catch (IOException e) {
        //     e.printStackTrace();
        //     System.out.println("Failed to save model: " + e.getMessage());
        // }

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

    // =====================================================================================================

    // public static void printLayerHelpers(ComputationGraph g, long[] inputShapeNHWC) {
    //     INDArray x = Nd4j.rand(inputShapeNHWC);

    //     g.output(false, x);
    //     Nd4j.getExecutioner().commit();

    //     System.out.println("=== Layer helper check (ComputationGraph) ===");
    //     for (org.deeplearning4j.nn.api.Layer layer : g.getLayers()) {
    //         Object helper = null;
    //         String foundIn = null;

    //         Class<?> c = layer.getClass();
    //         while (c != null) {
    //             try {
    //                 java.lang.reflect.Field f = c.getDeclaredField("helper");
    //                 f.setAccessible(true);
    //                 helper = f.get(layer);
    //                 foundIn = c.getName();
    //                 break;
    //             } catch (NoSuchFieldException e) {
    //                 c = c.getSuperclass();
    //             } catch (Throwable t) {
    //                 break;
    //             }
    //         }

    //         String helperName = (helper == null ? "null" : helper.getClass().getName());
    //         System.out.printf("%-40s helper=%s (field in %s)%n",
    //                 layer.conf().getLayer().getLayerName(),
    //                 helperName,
    //                 (foundIn == null ? "-" : foundIn));
    //     }

    //     x.close();
    // }
    // // =====================================================================================================

    // private static Object tryGetHelper(Layer l) {
    //     // Walk class hierarchy to find a field called "helper"
    //     Class<?> c = l.getClass();
    //     while (c != null) {
    //         try {
    //             Field f = c.getDeclaredField("helper");
    //             f.setAccessible(true);
    //             return f.get(l);
    //         } catch (NoSuchFieldException e) {
    //             c = c.getSuperclass();
    //         } catch (Throwable t) {
    //             return null;
    //         }
    //     }
    //     return null;
    // }

    // // =====================================================================================================

    // // public static void printLayerHelpers(MultiLayerNetwork net, long[] inputShape) {
    // //     // Example for CIFAR NCHW: new long[]{1, 3, 32, 32}
    // //     // Example for MNIST NCHW: new long[]{1, 1, 28, 28}
    // //     INDArray x = Nd4j.rand(inputShape);

    // //     net.output(x, false);
    // //     Nd4j.getExecutioner().commit();

    // //     System.out.println("=== Layer helper check (MultiLayerNetwork) ===");
    // //     for (Layer l : net.getLayers()) {
    // //         Object helper = tryGetHelper(l);
    // //         String helperName = (helper == null ? "null" : helper.getClass().getName());
    // //         System.out.printf("%-45s helper=%s%n",
    // //                 l.conf().getLayer().getLayerName(),
    // //                 helperName);
    // //     }

    // //     x.close();
    // // }

    // public static void printLayerHelpers(MultiLayerNetwork net, long[] inputShape) {
    //     INDArray x = Nd4j.rand(inputShape);

    //     net.output(x, false);
    //     Nd4j.getExecutioner().commit();

    //     System.out.println("=== Layer helper check (MultiLayerNetwork) ===");
    //     for (org.deeplearning4j.nn.api.Layer layer : net.getLayers()) {
    //         Object helper = null;
    //         String foundIn = null;

    //         Class<?> c = layer.getClass();
    //         while (c != null) {
    //             try {
    //                 Field f = c.getDeclaredField("helper");
    //                 f.setAccessible(true);
    //                 helper = f.get(layer);
    //                 foundIn = c.getName();
    //                 break;
    //             } catch (NoSuchFieldException e) {
    //                 c = c.getSuperclass();
    //             } catch (Throwable t) {
    //                 break;
    //             }
    //         }

    //         String helperName = (helper == null ? "null" : helper.getClass().getName());
    //         System.out.printf("%-35s helper=%s (field in %s)%n",
    //                 layer.conf().getLayer().getLayerName(),
    //                 helperName,
    //                 (foundIn == null ? "-" : foundIn));
    //     }

    //     x.close();
    // }
}

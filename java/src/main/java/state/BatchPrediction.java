
package state;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;

import utils.*;
import message.data_message.*; 
import message.weights_message.*; 

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;

public class BatchPrediction {

    private static final Config cfg = Config.getInstance();
    public final int NUM_FEATURES = cfg.NUM_FEATURES;
    public final int NUM_CLASSES = cfg.NUM_CLASSES;
    public final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;

    public final String DATASET = cfg.DATASET;
    private static final String LOSS_FUNCTION = cfg.LOSS_FUNCTION;
    private static final String COMBINE_LOSS = cfg.COMBINE_LOSS;
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT; 

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final MultiLayerNetwork model;
    private final MultiLayerNetwork bestModel;

    private static BatchPrediction coordinatorInstance = null;

    private final CustomLogger logger;
    private final boolean isCoordinator;

    private static boolean checked = false;
    private static double lambda = 1e-4;

    private final WorkerStatic ws;

    private long start = System.nanoTime();
    private long end = System.nanoTime();

    private final boolean MODEL_IS_CNN;
    private INDArray X4d;
    private INDArray X2d;
    private INDArray X;
    private INDArray Xbuffer;
    private INDArray probs; 
    private INDArray argMax;

    private int EXPECTED_SIZE; 
    private float EPS = 0.0000001f; 

    List<float[]> featureList = new ArrayList<>();
    List<Integer> labels = new ArrayList<>();
    float[] probabilities = new float[NUM_CLASSES];

    private final boolean coordinator;

    // for Worker (from WorkerStatic) =======================================================================================================

    public BatchPrediction(MultiLayerNetwork model, CustomLogger logger, WorkerStatic ws) {
        this.model = model;
        this.ws = ws;
        this.MODEL_IS_CNN = isCnnByFirstLayer(model);
        this.bestModel = null;
        this.logger = logger;
        this.isCoordinator = false;
        EXPECTED_SIZE = cfg.TRAIN_SIZE;

        if (MODEL_IS_CNN) {     // Instance Xbuffer based on nature / dimensionality of input data
            if ("mnist4".equals(DATASET) || "mnist".equals(DATASET)) {
                Xbuffer = Nd4j.create(EXPECTED_SIZE, 1, 28, 28);
            } else if ("cifar3".equals(DATASET)) {
                Xbuffer = Nd4j.create(EXPECTED_SIZE, 3, 32, 32);
            }
        } else {
            Xbuffer = Nd4j.create(EXPECTED_SIZE, NUM_FEATURES);
        }
        this.coordinator = false;
    }

    // for Coordinator ==================================================================================================

    public BatchPrediction(MultiLayerNetwork model, MultiLayerNetwork bestModel, CustomLogger logger) {
        this.model = model;
        this.ws = null;
        this.MODEL_IS_CNN = isCnnByFirstLayer(model);
        this.bestModel = bestModel;
        this.logger = logger;
        this.isCoordinator = true;
        this.EXPECTED_SIZE = 500;

        if (MODEL_IS_CNN) {
            if ("mnist4".equals(DATASET) || "mnist".equals(DATASET)) {
                Xbuffer = Nd4j.create(EXPECTED_SIZE, 1, 28, 28);
            } else if ("cifar3".equals(DATASET)) {
                Xbuffer = Nd4j.create(EXPECTED_SIZE, 3, 32, 32);
            }
        } else {
            Xbuffer = Nd4j.create(EXPECTED_SIZE, NUM_FEATURES);
        }

        this.coordinator = true;
    }

    // ===========================================================================

    public static BatchPrediction getInstanceForCoordinator(MultiLayerNetwork model, MultiLayerNetwork bestModel, CustomLogger logger) {
        if(coordinatorInstance == null) {
            coordinatorInstance = new BatchPrediction(model, bestModel, logger);
            return coordinatorInstance;
        } 
        return coordinatorInstance;
    }

    // ===========================================================================

    public static boolean isCnnByFirstLayer(MultiLayerNetwork model) {

        Layer l0 = model.getLayerWiseConfigurations().getConf(0).getLayer();
        return (l0 instanceof ConvolutionLayer); 
    }

    // ===========================================================================

    public float[] callPredictionsBatch(List<DataMessage> batch) {

        if (batch == null || batch.isEmpty()) {
            if (logger.isEnabled(2)) logger.log("Batch is empty");
            return null;
        }

        for (DataMessage msg : batch) {
            if (msg == null) continue;

            float[] features = msg.features;
            if (features == null) {
                if (logger.isEnabled(2)) logger.log("Null features in DataMessage");
                continue;
            }

            if (features.length != NUM_FEATURES) {
                if (logger.isEnabled(2)) logger.log("Wrong features length. Got " + features.length + " but NUM_FEATURES = " + NUM_FEATURES);
                System.out.println("Wrong features length. Got " + features.length + " but NUM_FEATURES = " + NUM_FEATURES);
                continue; // skip non-conforming record
            }

            featureList.add(features);
            labels.add(msg.label);
        }

        int nSamples = featureList.size();
        if (nSamples == 0) {
            if (logger.isEnabled(2)) logger.log("No samples after parsing batch");
            return null;
        }

        float[][] data = new float[nSamples][NUM_FEATURES];     // matrix of samples and features
        for (int i = 0; i < nSamples; i++) {
            System.arraycopy(featureList.get(i), 0, data[i], 0, NUM_FEATURES);
        }


        // Forward Pass Start ===============================================================================

        // ==============================================================================================================
        // Alternative 1) Costs Memory (Allocates new Memory every time), but Better Time and simplicity
        // doesnt seem to have a significant difference memory wise
        // choose this for better performance and simpler code
        // X2d = Nd4j.create(data); X = X2d.reshape(...); X = X4d.permute(...); happens on the CPU memory / host-side NDArray

        if(MODEL_IS_CNN) {
            if("cifar3".equals(DATASET)) {

                X2d = Nd4j.create(data);                       // [batch, 3072] => 3 * 32 * 32 = 3072
                X4d = X2d.reshape(nSamples, 32, 32, 3);        // [batch, 32, 32, 3]

                // if(checked == false) {
                //     float r = X4d.getFloat(0, 0, 0, 0);
                //     float g = X4d.getFloat(0, 0, 0, 1);
                //     float b = X4d.getFloat(0, 0, 0, 2);
                //     System.out.println("first pixel rgb = " + r + ", " + g + ", " + b);
                //     checked = true;
                // }

                // X = X4d.permute(0, 3, 1, 2).dup();        // rearange to => [batch, 3, 32, 32]
                X = X4d.permute(0, 3, 1, 2); 

            } else { // else if("mnist".equals(DATASET) || "mnist4".equals(DATASET) ) {

                X2d = Nd4j.create(data);          // [batch, 784]
                X = X2d.reshape(X2d.size(0), 1, 28, 28);
            }

        } else {    // Normal dataset (no image) + no CNN used 
            X = Nd4j.create(data);                     // [batch, NUM_FEATURES]
        }

        // ==============================================================================================================
        // Alternative 2) Costs Less Memory (Reuses / Overwrites the same buffer => Stable memory footprint), but costs more on Average Forward Pass Ms
        // Often much slower because scalar filling is the slowest possible way to build an INDArray (You call into ND4J once per element => goes Java → ND4J)
        // .create() is not “doing the same thing.” It’s doing it in one big vectorized move, not millions of function calls.

        // if (nSamples > EXPECTED_SIZE) {
        //     if (logger.isEnabled(2)) logger.log("Batch bigger than EXPECTED_SIZE: nSamples=" + nSamples + " EXPECTED_SIZE=" + EXPECTED_SIZE + " -> clipping");
        //     System.out.println("Batch bigger than EXPECTED_SIZE: nSamples=" + nSamples + " EXPECTED_SIZE=" + EXPECTED_SIZE + " -> clipping");

        //     nSamples = EXPECTED_SIZE;
        // }

        // for (int i = 0; i < nSamples; i++) {

        //     float[] features = featureList.get(i);

        //     if (MODEL_IS_CNN) {

        //         if ("mnist4".equals(DATASET) || "mnist".equals(DATASET)) {

        //             // flatten 784 into 1x28x28
        //             for (int j = 0; j < NUM_FEATURES; j++) {
        //                 int row = j / 28;
        //                 int col = j % 28;
        //                 Xbuffer.putScalar(new int[]{i, 0, row, col}, features[j]);
        //             }

        //         } else if ("cifar3".equals(DATASET)) {

        //             for (int j = 0; j < NUM_FEATURES; j++) {
        //                 int channel = j / (32 * 32);
        //                 int pixel = j % (32 * 32);
        //                 int row = pixel / 32;
        //                 int col = pixel % 32;

        //                 Xbuffer.putScalar(new int[]{i, channel, row, col}, features[j]);
        //             }
        //         }

        //     } else {
        //         for (int j = 0; j < NUM_FEATURES; j++) {
        //             Xbuffer.putScalar(i, j, features[j]);
        //         }
        //     }
        // }
        
        // if (nSamples == EXPECTED_SIZE) {
        //     X = Xbuffer;
        // } else {
        //     X = Xbuffer.get(
        //         NDArrayIndex.interval(0, nSamples),
        //         NDArrayIndex.all(),
        //         NDArrayIndex.all(),
        //         NDArrayIndex.all()
        //     );
        // }

        // ==============================================================================================================
        start = System.nanoTime();                // We only want to evaluate the performance of the forward pass, but this also includes the GPU transfer overhead
        probs = model.output(X, false);    // [batch, NUM_CLASSES] or [batch,1] if sigmoid. Here is where the memory transfer happens between CPU and GPU
                                                // this is one forward pass per batch (has multiple samples)
        end = System.nanoTime();
        // double min = probs.minNumber().doubleValue();
        // double max = probs.maxNumber().doubleValue();
        // System.out.println("probs min/max = " + min + " / " + max);

        // Forward Pass End ===============================================================================

        if (probs == null || probs.size(0) == 0) {
            if (logger.isEnabled(2)) logger.log("Empty probs batch");
            return null;
        }

        int nCorrect = 0;
        float loss = 0f;

        float[] sampleLosses = new float[nSamples];

        // BINARY CASE (SIGMOID) ======================================================================

        if (NEURAL_OUTPUT == 1) {

            for (int i = 0; i < nSamples; i++) {

                float p = probs.getFloat(i, 0);

                if (p < EPS) p = EPS;
                if (p > 1f - EPS) p = 1f - EPS;

                int label = labels.get(i);   // 0 or 1

                int pred = (p >= 0.5f) ? 1 : 0;
                if (pred == label) nCorrect++;

                sampleLosses[i] = LossFunction.compute_loss(p, label);
            }

        // MULTI-CLASS CASE (SOFTMAX) =================================================================

        } else {

            if(!LOSS_FUNCTION.equals("CROSS_ENTROPY") || true) {
                argMax = probs.argMax(1);   // max probability => this is what we are deciding
                
                float[] flatProps = probs.data().asFloat();  // converd 2D [nSamples, classes] into flat array
                
                for (int i = 0; i < nSamples; i++) {

                    int pred = argMax.getInt(i);
                    int label = labels.get(i);
                    if (pred == label) nCorrect++;

                    int base = i * NUM_CLASSES; 
                    for (int c = 0; c < NUM_CLASSES; c++) { // C order (row-major) - how it will implement
                        // 2D array to flat array conversion (here write 1 row, then 2 row, ...)
                                                            
                        probabilities[c] = flatProps[base + c];
                    }
                    sampleLosses[i] = LossFunction.compute_loss(probabilities, label);
                }

                // for (int i = 0; i < nSamples; i++) {

                //     int pred = argMax.getInt(i);
                //     int label = labels.get(i);
                //     if (pred == label) nCorrect++;

                //     // float[] probabilities = probs.getRow(i).toFloatVector();   
                //         // Expensive because: getRow(i) creates a view,
                //         // toFloatVector() allocates a new float[] and copies data every iteration
                //     // float p = probabilities[label];
                //     // if (p < EPS) p = EPS;

                //     // probabilities = probs.data().asFloat();
                //     INDArray row = probs.getRow(i);     // still creates a view per sample
                //     for (int c = 0; c < NUM_CLASSES; c++) {
                //         probabilities[c] = row.getFloat(c);     // doesnt allocate but overwrites memory
                //     }
                //     sampleLosses[i] = LossFunction.compute_loss(probabilities, label);
                    
                //     // float p1 = probs.getFloat(i, label);
                //     // logger.log("Label = " + label + ", probabilities: " + 
                //     //     Arrays.toString(probabilities) + ", p = " + p1);

                //     // if(p1 < EPS) {
                //     //     p1 = EPS;
                //     // }

                //     // sampleLosses[i] = (float) -Math.log(p1);
                // }

                // // logger.log("sampleLosses = " + Arrays.toString(sampleLosses) + 
                // //             "sampleLosses_fast = " + Arrays.toString(sampleLosses_fast));
                        
                // // logger.log("Same arrays? " + Arrays.equals(sampleLosses, sampleLosses_fast));


            } else {
                
                argMax = probs.argMax(1);

                for (int i = 0; i < nSamples; i++) {
                    int pred = argMax.getInt(i);
                    int label = labels.get(i);
                    if (pred == label) nCorrect++;

                    float p = probs.getFloat(i, label);     // probs.getFloat(i, j), probs is 2D-Array 
                                                            // probs = [nSamples, NUM_CLASSES]
                    if(p < EPS) p = EPS;
                    
                    sampleLosses[i] = (float) -Math.log(p); // cross-entropy
                }
            }
        }

        // Combine Losses from Multiple Samples =======================================================

        if ("TOP_K".equals(COMBINE_LOSS)) {
            loss = LossFunction.topKAverage(sampleLosses);

        } else if ("SUM".equals(COMBINE_LOSS)) {
            loss = LossFunction.sum(sampleLosses);

        } else if ("AVG".equals(COMBINE_LOSS)) {
            loss = LossFunction.average(sampleLosses);

        } else {
            loss = LossFunction.average(sampleLosses);
        }

        // Regularization =======================================================
        
        if(!this.coordinator) {
            double l2 = 0.0;    // add regularization
            for (float w : ws.flatModel) {   // or your flatWeights array
                l2 += w * w;
            }

            l2 /= ws.flatModel.length;
            loss = loss + (float)(lambda * l2);
        }

        // ===========================================================================================
        
        if (Float.isNaN(loss) || Float.isInfinite(loss)) {
            if (logger.isEnabled(2)) logger.log("loss is NaN/Inf, X length: " + X.length());
            return null;
        }

        if (nCorrect == 0) {
            if (logger.isEnabled(2)) logger.log("nCorrect == 0");
        }

        float accuracy = (float) nCorrect / nSamples;

        float forwardMs = (end - start) / 1_000_000f;
        // X.close();
        // probs.close();
        featureList.clear();
        labels.clear();
        return new float[]{accuracy, loss, nSamples, nCorrect, forwardMs};

        // } finally {
        //     if (probs != null) probs.close();
        //     if (X != null) X.close();
        //     if (argMax != null) argMax.close();
        //     if (X2d != null) X2d.close();
        //     if (X4d != null) X4d.close();
        // }

    }

        
    // ===========================================================================

    public String predictSingleBest(DataMessage msg) {
        if (msg == null || msg.features == null || msg.features.length != NUM_FEATURES) {
            return null;
        }

        try {
            INDArray X = Nd4j.create(msg.features).reshape(1, NUM_FEATURES);
            INDArray probs = bestModel.output(X, false);
            int pred = probs.argMax(1).getInt(0);

            Map<String, Object> out = new HashMap<>();
            out.put("sample_index", msg.sampleIndex);
            out.put("prediction", pred);

            return MAPPER.writeValueAsString(out);

        } catch (Exception e) {
            if (logger.isEnabled(2)) logger.log("Error in predictSingle: " + e.getMessage());
            return null;
        }
    }
}




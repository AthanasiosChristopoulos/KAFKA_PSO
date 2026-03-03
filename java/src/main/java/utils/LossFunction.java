package utils;

import java.util.Arrays;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.api.Layer;


public class LossFunction {

    private static final Config cfg = Config.getInstance();
    private static final String LOSS_FUNCTION = cfg.LOSS_FUNCTION;
    private static final int TOP_K_VALUE = cfg.TOP_K_VALUE;

    // =============================================================================================
    // Loss Function Control:

    public static float compute_loss(float[] probs, int label) {
        
        if ("MAE".equals(LOSS_FUNCTION) || "TOP_K".equals(cfg.COMBINE_LOSS)) {
            return compute_loss_MAE(probs, label);    

        } else if ("L2".equals(LOSS_FUNCTION)) {
            return compute_loss_MSE(probs, label);

        } else if ("CROSS_ENTROPY".equals(LOSS_FUNCTION)) {
            return compute_loss_CE(probs, label);

        } else if ("ZERO_ONE".equals(LOSS_FUNCTION)) {
            return compute_loss_zero_one(probs, label);
        }

        System.out.println("No valid loss function selected");
        return 10000f;
    }

    // =============================================================================================

    public static float compute_loss(float prob, int label) {

        if ("ZERO_ONE".equals(LOSS_FUNCTION)) {
            return compute_loss_zero_one_binary(prob, label);
        
        } else if("MAE".equals(LOSS_FUNCTION)) {
            return compute_loss_MAE_binary(prob, label);

        } else {
            return compute_loss_CE_binary(prob, label);
        
        }
    }

    // =============================================================================================

    public static float compute_loss_zero_one_binary(float prob, int label) {
        int pred = (prob >= 0.5f) ? 1 : 0;
        return (pred == label) ? 0f : 1f;
    }

    // =============================================================================================

    public static float compute_loss_MAE_binary(float prob, int label) {
        return Math.abs(prob - (float)label);
    }

    // =============================================================================================

    public static float compute_loss_CE_binary(float probs, int label) {  // binary cross entropy

        if (probs < 1e-7f) probs = 1e-7f;
        if (probs > 1f - 1e-7f) probs = 1f - 1e-7f;

        float loss = (float)(- (label * Math.log(probs) + (1 - label) * Math.log(1f - probs))); // Binary Cross Entropy Loss Function

        if (Float.isInfinite(loss)) {
            System.out.println("Sigmoid loss is Inf");
            return -1f;
        }

        if (Float.isNaN(loss)) {
            System.out.println("Sigmoid loss is NaN");
            return -1f;
        }

        return loss;
    }

    // =============================================================================================
    // Loss Functions:
    
    public static float compute_loss_MSE(float[] probs, int label) {

        if(probs.length == 0) {
            return -1f;
        }

        float[] target = new float[probs.length]; // convert to different one hot encoding: 2 => [0, 0, 1]
        target[label] = 1f;

        float sum = 0;
        for (int i = 0; i < probs.length; i++) {
                                    
            if (Float.isNaN(probs[i])) {
                System.out.println("probs[i] is NaN");
            }
            if (Float.isNaN(target[i])) {
                System.out.println("target[i] is NaN");
            }

            float d = probs[i] - target[i]; // probs[i] = [0, 1], when target[i] = y = [0 or 1]
            sum += d * d;   // L2, means squared
        }
        
        if (Float.isInfinite(sum)) {
            System.out.println("sum is Inf");
            return -1f;
        }

        if (Float.isNaN(sum)) {
            System.out.println("sum is NaN");
            return -1f;
        }

        return sum / probs.length;     // classical MSE , not half MSE (no need therea re no gradients in PSO)  
    }

    // =============================================================================================

    public static float compute_loss_CE(float[] probs, int label) {
        
        float eps = 0.0000001f;            
        float p = probs[label];

        if (p < eps) {          // avoid log(0)
            p = eps;
        }

        return (float) -Math.log(p);    // natural log; base doesn't really matter
    }
    
    // =============================================================================================
    // ZERO_ONE:

    public static float compute_loss_zero_one(float[] probs, int label) {

        if (probs == null || probs.length == 0) return -1f;

        int pred = 0;
        float best = probs[0];

        for (int c = 1; c < probs.length; c++) {    // compute y* = pred = argmax_c(p_c)
            float v = probs[c];
            if (v > best) {
                best = v;
                pred = c;
            }
        }

        return (pred == label) ? 0f : 1f;
    }

    // =============================================================================================
    // MAE / L1 loss between probs and one-hot target

    public static float compute_loss_MAE(float[] probs, int label) {
        if (probs == null || probs.length == 0) return -1f;

        float sum = 0f;
        int C = probs.length;
        float[] target = new float[probs.length];
        target[label] = 1f;     // on hot label encoding

        for (int c = 0; c < C; c++) {
            float d = target[c] - probs[c];
            sum += Math.abs(d);
        }

        return sum / C;
    }

    // =============================================================================================
    // Combination Functions:
    // =============================================================================================

    public static float sum(float[] sampleLosses) {
        if (sampleLosses == null || sampleLosses.length == 0) return 0f;

        float s = 0f;
        for (float v : sampleLosses) {
            s += v;
        }
        return s;
    }

    // =============================================================================================

    public static float average(float[] sampleLosses) {
        if (sampleLosses == null || sampleLosses.length == 0) return 0f;

        float s = sum(sampleLosses);
        return s / sampleLosses.length;
    }

    // =============================================================================================

    public static float topKAverage(float[] sampleLosses) {

        int n = sampleLosses.length;
        if (n == 0) return 0f;

        int k_edited = Math.min(TOP_K_VALUE, n);

        float[] tmp = Arrays.copyOf(sampleLosses, n);
        Arrays.sort(tmp);   // sorts in ascending order (min → max).

        float sum = 0f;
        for (int i = n - k_edited; i < n; i++) {    // we pick only the largest values (highest index)
            sum += tmp[i];
        }

        return sum / k_edited;
    }
    
    // ==================================================================================
    // Regularization - Weight Penalty Score
    // ==================================================================================
    // L2 - Penalty:

    public static double l2Penalty(float[] w) {
        if (w == null || w.length == 0) return 0.0;
        double s = 0.0;

        for (float v : w) {
            s += (double)v * (double)v;
        }
        return s / (double)w.length;
    }

    // ==================================================================================
    // Group-Lasso Penalty:

    public static double groupLassoNeuronPenalty(MultiLayerNetwork model, boolean includeBias) {

        double penalty = 0.0;

        for (int li = 0; li < model.getnLayers(); li++) {   // per Layer iterate
            
            Layer layer = model.getLayer(li);
            if (layer == null) continue;

            if (!layer.paramTable().containsKey("W")) continue; // if no weight

            INDArray W = layer.getParam("W");      // [nIn, nOut], these are all the weights of one layer
            if (W == null) continue;                        // [weight_inputs, neurons]

            INDArray b = null;
            if (includeBias && layer.paramTable().containsKey("b")) {
                b = layer.getParam("b");           // [nOut] or [1, nOut]
            }

            int nOut = (int) W.size(1);

            for (int j = 0; j < nOut; j++) {    // loop all the neurons

                INDArray col = W.getColumn(j);  // per one neuron, select all the incoming weights
                                                // col[nIn] (has the dimensionality of the inputs)
                double sumSq = col.mul(col).sumNumber().doubleValue();
                            // multiple all weights with themselves and then add them together (sumNumber)
                if (b != null) {
                    double bj = b.getDouble(j);
                    sumSq += bj * bj;
                }

                penalty += Math.sqrt(sumSq);       // ||group||_2
            }
        }

        return penalty;
    }

    // ==================================================================================
    // slope Penalty 

    public static double slopePenalty(float[] w, float[] lambdas) {

        if (w == null || w.length == 0) return 0.0;

        int d = w.length;
        if (lambdas == null || lambdas.length < d) {
            throw new IllegalArgumentException("SLOPE lambdas length must be >= #weights (need " + d + ")");
        }

        float[] weights_abs = new float[d];
        for (int i = 0; i < d; i++) {
            float v = w[i];
            weights_abs[i] = (v >= 0f) ? v : -v;
        }

        Arrays.sort(weights_abs); // ascending

        double pen = 0.0;
        int j = 0;      // lambda_1 applies to largest |w|

        for (int idx = d - 1; idx >= 0; idx--) {
            pen += (double) lambdas[j] * (double) weights_abs[idx];
            j++;
        }

        return pen;
    }
    
    // =============================================================================================
    // Based on geometric decay λj​=λ1​αj−1,  0<α≤1
    
    public static float[] makeSlopeLambdasGeometric(int d, float lambda1, float alpha) {
        if (d <= 0) return new float[0];
        if (alpha <= 0f || alpha > 1f) throw new IllegalArgumentException("alpha must be in (0, 1].");

        float[] l = new float[d];
        float cur = lambda1;
        for (int j = 0; j < d; j++) {
            l[j] = cur;
            cur *= alpha;
        }
        return l;
    }
}

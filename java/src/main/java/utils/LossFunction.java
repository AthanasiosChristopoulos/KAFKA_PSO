package utils;

import java.util.Arrays;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.api.Layer;


public class LossFunction {

    private static final Config cfg = Config.getInstance();

    // =============================================================================================
    // Loss Function Control:

    public static float compute_loss(float[] probs, int label) {
        if ("MAE".equals(cfg.LOSS_FUNCTION)) {
            return compute_loss_MAE(probs, label);    

        } else if ("L2".equals(cfg.LOSS_FUNCTION)) {
            return compute_loss_MSE(probs, label);

        } else if ("CROSS_ENTROPY".equals(cfg.LOSS_FUNCTION)) {
            return compute_loss_CE(probs, label);

        } else if ("ZERO_ONE".equals(cfg.LOSS_FUNCTION)) {
            return compute_loss_zero_one(probs, label);

        } else if ("ABSOLUTE_MARGIN".equals(cfg.LOSS_FUNCTION)) {
            return compute_loss_margin_abs(probs, label);

        } else if ("HINGE".equals(cfg.LOSS_FUNCTION)) {
            // System.out.println("HINGE");
            return compute_loss_hinge(probs, label);
            // return compute_loss_topk_hinge(probs, label, 5);

        } else if ("RAMP".equals(cfg.LOSS_FUNCTION)) {
            // System.out.println("AAAAAAAAAAAAAAAAAAAA");
            return compute_loss_ramp(probs, label);
        }

        System.out.println("No valid loss function selected");
        return 10000f;
    }

    // =============================================================================================

    public static float compute_loss(float prob, int label) {

        if ("ZERO_ONE".equals(cfg.LOSS_FUNCTION)) {
            return compute_loss_zero_one_binary(prob, label);
        
        } else if("MAE".equals(cfg.LOSS_FUNCTION)) {
            return compute_loss_MAE_binary(prob, label);

        } else if("HINGE".equals(cfg.LOSS_FUNCTION)) {
            return compute_loss_hinge_binary(prob, label);

        } else if("RAMP".equals(cfg.LOSS_FUNCTION)) {
            return compute_loss_ramp_binary(prob, label);

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
    // Non Differentiable Functions
    // =============================================================================================
    // Hinge Loss

    public static float compute_loss_hinge_binary(float prob, int label) {

        // convert label {0,1} → {-1,+1}
        int y = (label == 1) ? 1 : -1;

        // convert probability to score [-1,1]
        float score = 2f * prob - 1f;

        // hinge loss: L = max(0, 1 - y f(x))
        return Math.max(0f, 1f - y * score);
    }

    // =============================================================================================
    // top-1 Hinge
    public static float compute_loss_hinge(float[] scores, int label) {  

        float trueScore = scores[label];    // trueScore
        float maxOther = -Float.MAX_VALUE;

        // L=max(0,1−ys)
        // If 𝑦𝑠≥1 inside is ≤ 0 → loss = 0
        // Meaning: “this example is already good enough; don’t waste effort making it even larger.”
        // If ys<1: inside is positive → loss grows linearly (the more wrong the quess was)
        // Meaning: “penalize violations of the margin.”

        for (int i = 0; i < scores.length; i++) {
            if (i == label) continue;
            if (scores[i] > maxOther) maxOther = scores[i];
        }

        float margin = trueScore - maxOther;
        return Math.max(0f, 1f - margin);   // once the prediction is confident enough, no more penalty
    }

    // =============================================================================================
    // top-k Hinge
    public static float compute_loss_topk_hinge(float[] scores, int label, int k) {
        // System.out.println("AAAA");
        int C = scores.length;
        float trueScore = scores[label];

        float[] violations = new float[C - 1];
        int idx = 0;

        for (int j = 0; j < C; j++) {
            if (j == label) continue;
            violations[idx++] = 1f + scores[j] - trueScore;
        }

        Arrays.sort(violations); // ascending
        int kk = Math.min(k, violations.length);

        float sum = 0f;
        for (int i = violations.length - kk; i < violations.length; i++) {
            sum += violations[i];
        }

        return Math.max(0f, sum / kk);
    }
    // =============================================================================================
    // ε-insensitive loss

    public static float compute_loss_epsilon_insensitive_binary(float prob, int label) {
        float eps = 0.1f; // tune this
        float target = (float) label;   // label in {0,1}
        float err = Math.abs(prob - target);
        return Math.max(0f, err - eps);
    }

    // =============================================================================================

    public static float compute_loss_epsilon_insensitive(float[] probs, int label) {
        float eps = 0.1f; // tune this

        if (probs == null || probs.length == 0) return -1f;

        float sum = 0f;
        int C = probs.length;

        for (int c = 0; c < C; c++) {
            float target = (c == label) ? 1f : 0f;
            float err = Math.abs(probs[c] - target);
            sum += Math.max(0f, err - eps);
        }

        return sum / C;
    }

    // =============================================================================================
    // Ramp Loss

    public static float compute_loss_ramp_binary(float prob, int label) {
        float marginTarget = 1f;

        // label {0,1} -> {-1,+1}
        int y = (label == 1) ? 1 : -1;

        // probability [0,1] -> score [-1,1]
        float score = 2f * prob - 1f;

        float z = marginTarget - y * score;

        return Math.max(0f, Math.min(1f, z));
    }

    // =============================================================================================

    public static float compute_loss_ramp(float[] probs, int label) {
        float marginTarget = 1f;

        if (probs == null || probs.length == 0) return -1f;

        float py = probs[label];
        float maxOther = -Float.MAX_VALUE;

        for (int i = 0; i < probs.length; i++) {
            if (i == label) continue;
            if (probs[i] > maxOther) maxOther = probs[i];
        }

        // multiclass margin analogue
        float margin = py - maxOther;
        float z = marginTarget - margin;

        return Math.max(0f, Math.min(1f, z));
    }

    // =============================================================================================
    // Differentiable Functions
    // =============================================================================================
    // MSE

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
    
    // ============================================================================================
    // Non - Differentiable
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
    // Absolute Margin Loss
    // non-diff at margin = 1 because max switches branch.

    public static float compute_loss_margin_abs(float[] probs, int label) { // is non differentiable at 0
        // penalizes margin violations (largest wrong class probability)
        float py = probs[label];

        float maxOther = -Float.MAX_VALUE;

        for (int i = 0; i < probs.length; i++) {
            if (i == label) continue;   // this is why its maxOther. Its the max probability other than the one with the label
            if (probs[i] > maxOther) maxOther = probs[i];
        }

        // Implement L= ∣1 − (py​−pmax_other​)∣
        float margin = py - maxOther;   // if the margin grows larger than 1, the loss increases again.
        return Math.abs(1f - margin);   // too small margins, too large margins
                                        // margin ≈ 1
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

        int k_edited = Math.min(cfg.TOP_K_VALUE, n);

        float[] tmp = Arrays.copyOf(sampleLosses, n);
        Arrays.sort(tmp);   // sorts in ascending order (min → max).

        float sum = 0f;
        for (int i = n - k_edited; i < n; i++) {    // we pick only the largest values (highest index)
            sum += tmp[i];
        }

        return sum / k_edited;
    }
    
    // ==================================================================================
    // Regularization
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
            weights_abs[i] = (v >= 0f) ? v : -v;    // compute |w| element wise == weight_abs
        }

        Arrays.sort(weights_abs); // sort weight abs, ascending

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
            cur *= alpha;   // defined based on geometric decay, accumulationg this alpha multiplier,
                            // which is smaller than 1 (0<alpha≤1)
        }
        return l;
    }
}

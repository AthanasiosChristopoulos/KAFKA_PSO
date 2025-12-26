package utils;

import java.util.Arrays;


public class LossFunction {

    private static final Config cfg = Config.getInstance();
    private static final String LOSS_FUNCTION = cfg.LOSS_FUNCTION;
    private static final int TOP_K_VALUE = cfg.TOP_K_VALUE;

    // =============================================================================================
    // Loss Function Control:

    public static float compute_loss(float[] probs, int label) {
        
        if ("L2".equals(LOSS_FUNCTION)) {
            return compute_loss_MSE(probs, label);

        } else if ("CROSS_ENTROPY".equals(LOSS_FUNCTION)) {
            return compute_loss_CE(probs, label);
        }

        System.out.println("No valid loss function selected");
        return 10000f;
    }

    // =============================================================================================

    public static float compute_loss(float probs, int label) {
        return compute_loss_binary_CE(probs, label);

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

    public static float compute_loss_binary_CE(float probs, int label) {  // binary cross entropy

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

    public static float compute_loss_CE(float[] probs, int label) {

        float eps = 0.0000001f;            
        float p = probs[label];

        if (p < eps) {          // avoid log(0)
            p = eps;
        }

        return (float) -Math.log(p);    // natural log; base doesn't really matter
    }
    
    // =============================================================================================
    // Combination Functions:

    public static float sum(float[] sampleLosses) {
        if (sampleLosses == null || sampleLosses.length == 0) return 0f;

        float s = 0f;
        for (float v : sampleLosses) {
            // optionally ignore bad sentinel values
            if (Float.isNaN(v) || Float.isInfinite(v)) return Float.POSITIVE_INFINITY;
            s += v;
        }
        return s;
    }

    // =============================================================================================

    public static float average(float[] sampleLosses) {
        if (sampleLosses == null || sampleLosses.length == 0) return 0f;

        float s = sum(sampleLosses);
        if (Float.isInfinite(s)) return s; // propagate infinity
        return s / sampleLosses.length;
    }

    // =============================================================================================

    public static float topKAverage(float[] sampleLosses) {

        int n = sampleLosses.length;
        if (n == 0) return 0f;

        int k_edited = Math.min(TOP_K_VALUE, n);

        float[] tmp = Arrays.copyOf(sampleLosses, n);
        Arrays.sort(tmp);

        float sum = 0f;
        for (int i = n - k_edited; i < n; i++) {
            sum += tmp[i];
        }

        return sum / k_edited;
    }
    

    // public static float compute_loss_top_k(float[] probs, int label) {

    //     int n = probs.length;
    //     if (n == 0) return 0f;

    //     int k_value = Math.min(k, n);
        
    //     float[] target = new float[n];
    //     target[label] = 1f;

    //     float[] absResiduals = new float[n];
    //     for (int i = 0; i < n; i++) {   // calculate all the residuals (k independent)
    //         absResiduals[i] = Math.abs(probs[i] - target[i]);   // the magnitudes
    //     }

    //     Arrays.sort(absResiduals);  // sort ascending order

    //     if(sampled == false) {
    //         System.out.println("Sample Sorted: " + Arrays.toString(absResiduals));
    //         sampled = true;
    //     }

    //     float sumTopK = 0f;
    //     for (int i = n - k_value; i < n; i++) {
    //         sumTopK += absResiduals[i];
    //     }

    //     return sumTopK / k_value;
    // }
}

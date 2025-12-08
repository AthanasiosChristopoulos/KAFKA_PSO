package utils;

import java.util.Arrays;


public class LossFunction {

    private static final Config cfg = Config.getInstance();
    private static final String LOSS_FUNCTION = cfg.LOSS_FUNCTION;
    private static final int k = cfg.TOP_K_VALUE;

    public static float compute_loss(float[] probs, int label) {
        
        if ("L2".equals(LOSS_FUNCTION)) {
            return compute_loss_L2(probs, label);

        } else if ("TOP_K".equals(LOSS_FUNCTION)) {
            return compute_loss_top_k(probs, label);

        } else if ("CROSS_ENTROPY".equals(LOSS_FUNCTION)) {
            return compute_loss_CE(probs, label);
        }

        System.out.println("No valid loss function selected");
        return 10000f;
    }

    // =============================================================================================

    public static float compute_loss_L2(float[] probs, int label) {

        if(probs.length == 0) {
            return -1f;
        }

        float[] target = new float[probs.length]; // convert to different one hot encoding: 2 => [0, 0, 1]
        target[label] = 1f;

        float sum = 0;
        for (int i = 0; i < probs.length; i++) {
            // System.out.println("Probs: " +  probs[i] + ", target: " + target[i]);
                        
            if (Float.isNaN(probs[i])) {
                System.out.println("probs[i] is NaN");
            }
            if (Float.isNaN(target[i])) {
                System.out.println("target[i] is NaN");
            }

            float d = probs[i] - target[i];
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
        // if(sum == 0) {
        //     System.out.println("Sum is 0");
        //     for (int i = 0; i < probs.length; i++) {
        //         System.out.println("Probs: " +  probs[i] + ", target: " + target[i]);
        //     }
        // }

        return 0.5f * sum;     // classical MSE    
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

    public static float compute_loss_top_k(float[] probs, int label) {

        int n = probs.length;
        if (n == 0) return 0f;

        int k_value = Math.min(k, n);
        // System.out.println("K: " + K);
        
        float[] target = new float[n];
        target[label] = 1f;

        float[] absResiduals = new float[n];
        for (int i = 0; i < n; i++) {   // calculate all the residuals (k independent)
            absResiduals[i] = Math.abs(probs[i] - target[i]);
        }

        Arrays.sort(absResiduals);      // sort them

        float sumTopK = 0f;
        for (int i = n - k_value; i < n; i++) {
            sumTopK += absResiduals[i];
        }

        return sumTopK / k_value;
    }
}

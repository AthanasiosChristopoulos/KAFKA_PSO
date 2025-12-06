package utils;

import java.util.Arrays;


public class LossFunction {

    private static final Config cfg = Config.getInstance();
    private static final String LOSS_FUNCTION = cfg.LOSS_FUNCTION;
    private static final int k = cfg.TOP_K_VALUE;

    public static double compute_loss(double[] probs, int label) {
        
        if ("L2".equals(LOSS_FUNCTION)) {
            return compute_loss_L2(probs, label);

        } else if ("TOP_K".equals(LOSS_FUNCTION)) {
            return compute_loss_top_k(probs, label);

        } else if ("CROSS_ENTROPY".equals(LOSS_FUNCTION)) {
            return compute_loss_CE(probs, label);
        }

        System.out.println("No valid loss function selected");
        return 10000.0;
    }

    // =============================================================================================

    public static double compute_loss_L2(double[] probs, int label) {

        double[] target = new double[probs.length]; // convert to different one hot encoding: 2 => [0, 0, 1]
        target[label] = 1.0;

        double sum = 0;
        for (int i = 0; i < probs.length; i++) {
            double d = probs[i] - target[i];
            sum += d * d;   // L2, means squared
        }

        return 0.5 * sum;     // classical MSE    
    }

    // =============================================================================================

    public static double compute_loss_CE(double[] probs, int label) {

        double eps = 1e-12;            
        double p = probs[label];

        if (p < eps) {          // avoid log(0)
            p = eps;
        }

        return -Math.log(p);    // natural log; base doesn't really matter
    }
    
    // =============================================================================================

    public static double compute_loss_top_k(double[] probs, int label) {

        int n = probs.length;
        if (n == 0) return 0.0;

        int k_value = Math.min(k, n);
        // System.out.println("K: " + K);
        
        double[] target = new double[n];
        target[label] = 1.0;

        double[] absResiduals = new double[n];
        for (int i = 0; i < n; i++) {   // calculate all the residuals (k independent)
            absResiduals[i] = Math.abs(probs[i] - target[i]);
        }

        Arrays.sort(absResiduals);      // sort them

        double sumTopK = 0.0;
        for (int i = n - k_value; i < n; i++) {
            sumTopK += absResiduals[i];
        }

        return sumTopK / k_value;
    }
}

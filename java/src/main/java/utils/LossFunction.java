package utils;

import java.util.Arrays;


public class LossFunction {

    private static final Config cfg = Config.get();
    private static final String LOSS_FUNCTION = cfg.LOSS_FUNCTION;
    private static final int k = cfg.TOP_K_VALUE;

    public static double compute(double[] probs, int label) {
        if ("L2".equals(LOSS_FUNCTION)) {
            return compute_L2(probs, label);
        } else if ("TOP_K".equals(LOSS_FUNCTION)) {
            return compute_top_k(probs, label);
        }
        return 10000.0;
    }

    public static double compute_L2(double[] probs, int label) {
        double[] target = new double[probs.length];
        target[label] = 1.0;

        double sum = 0;
        for (int i = 0; i < probs.length; i++) {
            double d = probs[i] - target[i];
            sum += d * d;
        }
        return sum;
    }

    public static double compute_top_k(double[] probs, int label) {
        int n = probs.length;
        if (n == 0) return 0.0;

        int kk = Math.min(k, n);

        double[] target = new double[n];
        target[label] = 1.0;

        double[] absResiduals = new double[n];
        for (int i = 0; i < n; i++) {
            absResiduals[i] = Math.abs(probs[i] - target[i]);
        }

        Arrays.sort(absResiduals);

        double sumTopK = 0.0;
        for (int i = n - kk; i < n; i++) {
            sumTopK += absResiduals[i];
        }

        return sumTopK / kk;
    }
}

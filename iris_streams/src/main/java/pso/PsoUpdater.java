package pso;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

public class PsoUpdater {

    private Logger LOG;

    private final double W_INERTIA;
    private final double C1;
    private final double C2;

    private final int NUM_WORKERS;

    private double[] velocity; 

    public PsoUpdater(MultiLayerNetwork model, int workerId) {
        
        // W_INERTIA = Double.parseDouble(System.getenv().getOrDefault("W_INERTIA", "0.7"));
        W_INERTIA = 0.7;
        C1 = Double.parseDouble(System.getenv().getOrDefault("C1", "1.229"));
        C2 = Double.parseDouble(System.getenv().getOrDefault("C2", "1.229"));

        NUM_WORKERS = Integer.parseInt(System.getenv().getOrDefault("NUM_WORKERS", "10"));

        double[] x = Dl4jParamUtils.modelToFlatList(model);
        velocity = new double[x.length];

        randomizeVelocity(workerId, 0.01);
        // randomizeModelWeights(model, workerId + 1337, 0.1);
    }

    //================================================================================================

    public double[] updateX(MultiLayerNetwork model, double[] pbest, double[] gbest) {

        // Current position x_i
        double[] x_i = Dl4jParamUtils.modelToFlatList(model);
        int dim = x_i.length;

        // If velocity not initialized, initialize with zeros
        if (velocity == null || velocity.length != dim) {
            velocity = new double[dim];
        }

        Random rnd = new Random();

        double[] velocity_i_1 = new double[dim];
        double[] x_i_1 = new double[dim];

        for (int k = 0; k < dim; k++) {

            double r1 = rnd.nextDouble();   // randomness
            double r2 = rnd.nextDouble();  

            double cognitive = C1 * r1 * (pbest[k] - x_i[k]);
            double social    = C2 * r2 * (gbest[k] - x_i[k]);
            double inertia   = W_INERTIA * velocity[k];

            velocity_i_1[k] = inertia + cognitive + social;
            x_i_1[k] = x_i[k] + velocity_i_1[k];
        }

        // Update model parameters with x_i_1
        Dl4jParamUtils.updateModel(model, x_i_1);

        // Store velocity and return it
        this.velocity = velocity_i_1;
        return velocity_i_1;
    }

    //================================================================================================

    // public double[] updateX(MultiLayerNetwork model, List<double[]> neighborPBestList) {

    //     double[] x_i = Dl4jParamUtils.modelToFlatList(model);

    //     double[] socialAggregate = new double[x_i.length];
    //     Random rnd = new Random();

    //     // for pBest_j in neighbor_pBests:
    //     for (double[] pBest_j : neighborPBestList) {

    //         if (pBest_j.length != x_i.length) {
    //             throw new IllegalArgumentException("pBest size mismatch");
    //         }

    //         for (int k = 0; k < x_i.length; k++) {
    //             double p_i_j = rnd.nextDouble();  // in [0,1)
    //             socialAggregate[k] += p_i_j * (pBest_j[k] - x_i[k]);
    //         }
    //     }

    //     double scale = C_SOCIAL / (double) NUM_WORKERS;
    //     for (int k = 0; k < socialAggregate.length; k++) {
    //         socialAggregate[k] *= scale;
    //     }

    //     double[] velocity_i_1 = new double[x_i.length];
    //     double[] x_i_1 = new double[x_i.length];

    //     for (int k = 0; k < x_i.length; k++) {
    //         velocity_i_1[k] = W_INERTIA * velocity[k] + socialAggregate[k];
    //         x_i_1[k] = x_i[k] + velocity_i_1[k];
    //     }

    //     // update model
    //     Dl4jParamUtils.updateModel(model, x_i_1);

    //     // this.LOG.info("New model params sample: " + sampleFlat(x_i_1, 5));

    //     this.velocity = velocity_i_1;
    //     return this.velocity;
    // }

    //================================================================================================

    public void randomizeVelocity(int workerId, double sigma) {
        Random rnd = new Random(workerId);

        for (int i = 0; i < velocity.length; i++) {
            velocity[i] = rnd.nextGaussian() * sigma;
        }

        // this.LOG.info("Velocity initialized: " + sampleFlat(velocity, 5));
    }

    public void randomizeModelWeights(MultiLayerNetwork model, int seed, double sigma) {
        double[] flat = Dl4jParamUtils.modelToFlatList(model);
        Random rnd = new Random(seed);

        for (int i = 0; i < flat.length; i++) {
            flat[i] += rnd.nextGaussian() * sigma;
        }

        Dl4jParamUtils.updateModel(model, flat);

        // this.LOG.info("Model randomized: " + sampleFlat(flat, 5));
    }

    private static String sampleFlat(double[] flat, int n) {
        StringBuilder sb = new StringBuilder("[");
        int len = Math.min(n, flat.length);
        for (int i = 0; i < len; i++) {
            sb.append(String.format("%.4f", flat[i]));
            if (i < len - 1) sb.append(", ");
        }
        if (flat.length > n) sb.append(", ...");
        sb.append("]");
        return sb.toString();
    }
}

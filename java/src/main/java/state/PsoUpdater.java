package state;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.List;
import java.util.Random;

import utils.*; 

public class PsoUpdater {

    private final float W_INERTIA;
    private final float C;
    private final float C1;
    private final float C2;
    private final int NUM_WORKERS;
    private final boolean FULLY_INFORMED;

    private float[] velocity; 

    public PsoUpdater(MultiLayerNetwork model, int workerId) {
        
        Config cfg = Config.getInstance();
        this.FULLY_INFORMED = cfg.FULLY_INFORMED;

        if(FULLY_INFORMED == true) {
            this.W_INERTIA = cfg.W_INERTIA;
        } else {
           this.W_INERTIA = cfg.W_INERTIA_G_BEST;
        }

        this.C = cfg.C;
        this.C1 = cfg.C1;
        this.C2 = cfg.C2;
        this.NUM_WORKERS = cfg.NUM_WORKERS;

        float[] x = Dl4jParamUtils.modelToFlatList(model);
        velocity = new float[x.length];

        randomizeVelocity(workerId, 0.01f);
    }

    //================================================================================================

    public float[] updateX(MultiLayerNetwork model, float[] pbest, float[] gbest) {

        float[] x_i = Dl4jParamUtils.modelToFlatList(model);
        int dim = x_i.length;

        if (velocity == null || velocity.length != dim) {
            velocity = new float[dim];
        }

        Random rnd = new Random();

        float[] velocity_i_1 = new float[dim];
        float[] x_i_1 = new float[dim];

        for (int k = 0; k < dim; k++) {

            float r1 = rnd.nextFloat();   // randomness
            float r2 = rnd.nextFloat();  

            float cognitive = C1 * r1 * (pbest[k] - x_i[k]);
            float social    = C2 * r2 * (gbest[k] - x_i[k]);
            float inertia   = W_INERTIA * velocity[k];

            velocity_i_1[k] = inertia + cognitive + social;
            x_i_1[k] = x_i[k] + velocity_i_1[k];
        }

        Dl4jParamUtils.updateModel(model, x_i_1);

        this.velocity = velocity_i_1;
        return velocity_i_1;
    }

    //================================================================================================

    // public float[] updateX(MultiLayerNetwork model, List<float[]> neighborPBestList) {

    //     float[] x_i = Dl4jParamUtils.modelToFlatList(model);

    //     float[] socialAggregate = new float[x_i.length];
    //     Random rnd = new Random();

    //     // for pBest_j in neighbor_pBests:
    //     for (float[] pBest_j : neighborPBestList) {

    //         if (pBest_j.length != x_i.length) {
    //             throw new IllegalArgumentException("pBest size mismatch");
    //         }

    //         for (int k = 0; k < x_i.length; k++) {
    //             float p_i_j = rnd.nextFloat();  // in [0,1)
    //             socialAggregate[k] += p_i_j * (pBest_j[k] - x_i[k]);
    //         }
    //     }

    //     float scale = C / (float) NUM_WORKERS;
    //     for (int k = 0; k < socialAggregate.length; k++) {
    //         socialAggregate[k] *= scale;
    //     }

    //     float[] velocity_i_1 = new float[x_i.length];
    //     float[] x_i_1 = new float[x_i.length];

    //     for (int k = 0; k < x_i.length; k++) {
    //         velocity_i_1[k] = W_INERTIA * velocity[k] + socialAggregate[k];
    //         x_i_1[k] = x_i[k] + velocity_i_1[k];
    //     }

    //     Dl4jParamUtils.updateModel(model, x_i_1);

    //     this.velocity = velocity_i_1;
    //     return this.velocity;
    // }

    public float[] updateX(MultiLayerNetwork model, List<float[]> neighborPBestList) {

        float[] x_i = Dl4jParamUtils.modelToFlatList(model);
        float[] socialAggregate = new float[x_i.length];
        Random rnd = new Random();

        // ===== handle "no neighbors" case: pure inertia step =====
        if (neighborPBestList == null || neighborPBestList.isEmpty()) {
            float[] velocity_i_1 = new float[x_i.length];
            float[] x_i_1 = new float[x_i.length];

            for (int k = 0; k < x_i.length; k++) {
                // socialAggregate[k] is 0 -> only inertia
                velocity_i_1[k] = W_INERTIA * velocity[k];
                x_i_1[k] = x_i[k] + velocity_i_1[k];
            }

            Dl4jParamUtils.updateModel(model, x_i_1);
            this.velocity = velocity_i_1;
            return this.velocity;
        }

        // ===== normal fully-informed case with neighbors =====
        // for pBest_j in neighbor_pBests:
        for (float[] pBest_j : neighborPBestList) {

            if (pBest_j.length != x_i.length) {
                throw new IllegalArgumentException("pBest size mismatch");
            }

            for (int k = 0; k < x_i.length; k++) {
                float p_i_j = rnd.nextFloat();  // in [0,1)
                socialAggregate[k] += p_i_j * (pBest_j[k] - x_i[k]);
            }
        }

        float scale = C / (float) NUM_WORKERS;
        for (int k = 0; k < socialAggregate.length; k++) {
            socialAggregate[k] *= scale;
        }

        float[] velocity_i_1 = new float[x_i.length];
        float[] x_i_1 = new float[x_i.length];

        for (int k = 0; k < x_i.length; k++) {
            velocity_i_1[k] = W_INERTIA * velocity[k] + socialAggregate[k];
            x_i_1[k] = x_i[k] + velocity_i_1[k];
        }

        Dl4jParamUtils.updateModel(model, x_i_1);

        this.velocity = velocity_i_1;
        return this.velocity;
    }

    //================================================================================================

    public void randomizeVelocity(int workerId, float sigma) {
        Random rnd = new Random(workerId);

        for (int i = 0; i < velocity.length; i++) {
            velocity[i] = (float) rnd.nextGaussian() * sigma;
        }
    }

    //================================================================================================

    public void randomizeModelWeights(MultiLayerNetwork model, int seed, float sigma) {
        float[] flat = Dl4jParamUtils.modelToFlatList(model);
        Random rnd = new Random(seed);

        for (int i = 0; i < flat.length; i++) {
            flat[i] += rnd.nextGaussian() * sigma;
        }

        Dl4jParamUtils.updateModel(model, flat);
    }

}

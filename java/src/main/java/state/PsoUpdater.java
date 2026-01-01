package state;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.List;
import java.util.Random;
import java.util.Arrays;

import utils.*; 

public class PsoUpdater {

    private Config cfg = Config.getInstance();
    private final float W_INERTIA = cfg.W_INERTIA;;
    private final float C = cfg.C;
    private final float C1 = cfg.C1;
    private final float C2 = cfg.C2;
    private final int N_WORKERS = cfg.N_WORKERS;

    private final float C1_START = cfg.C1; 
    private final float C1_END = 0.2f;
    private float c1 = C1_START;  
    private int iter = 0;
    private final int MAX_ITERS = 500;
    private final int C1_MID_UPDATE = 600;   // sigmoid midpoint (where it drops fastest)
    private final int C1_MAX_UPDATES = 40 * 10000 / N_WORKERS; // expected max updates (for clamping)
    private final float C1_DROP_WIDTH = 200f;   // the 200 means “mostly drops between 600±100” → around 500–700

    private final float VMAX;    
    private final float VMAX_FACTOR;
    private final float VMAX_NORM;

    private CustomLogger logger;
    private float[] velocity; 
    private float[] x_i_new;
    private float[] inertiaVec;
    private float[] cognitiveVec;
    private float[] socialVec;
    private float[] diffPBestGBest;

    private int clamp_count = 0;

    private int count_updates = 0;

    public PsoUpdater(MultiLayerNetwork model, int workerId) {
    
        float[] x = Dl4jParamUtils.modelToFlatList(model);
        x_i_new = new float[x.length];
        velocity = new float[x.length];
        inertiaVec = new float[x.length];
        cognitiveVec = new float[x.length];
        socialVec = new float[x.length];
        diffPBestGBest = new float[x.length];

        this.VMAX_FACTOR = cfg.VMAX_FACTOR;

        float xmin = -1.0f; // Each individual weight is allowed to change at this rate
        float xmax = 1.0f;  // During training, most weights should stay relatively small (in practice < 0.2 or < 0.5).

        float range = xmax - xmin;  // the xmax - xmin discussed in the paper 

        this.VMAX = VMAX_FACTOR * range;  // VMAX_FACTOR == the δ discussed in the paper 
        this.VMAX_NORM = (float)(Math.sqrt(x.length) * VMAX);

        randomizeVelocity(workerId, 0.1f); //  0.1f this affects the magnitude of the initialized velocity

        this.logger = CustomLogger.getWorkerInstance(workerId);

        logger.log("Number of weights (dimensionality): " + x.length);
    }

    //================================================================================================

    private float clampVelocity(float v) {

        if (v > VMAX) {
            clamp_count++;
            return VMAX;
        }
        
        if (v < -VMAX) {
            clamp_count++;
            return -VMAX;
        }

        return v;
    }

    //================================================================================================

    private void clipVelocityByNorm(float vmaxNorm) {
        double sumSq = 0.0;
        for (float v : velocity) sumSq += (double)v * v;
        double norm = Math.sqrt(sumSq);

        if (norm > vmaxNorm && norm > 0.0) {
            float scale = (float)(vmaxNorm / norm);
            for (int i = 0; i < velocity.length; i++) velocity[i] *= scale;
        }
    }

    //================================================================================================

    public float[] updateX(MultiLayerNetwork model, float[] pbest, float[] gbest) {     // FOR GBEST, not fully informed

        count_updates++;

        float[] x_i = Dl4jParamUtils.modelToFlatList(model);
        clamp_count = 0;
        // if (velocity == null || velocity.length != x_i.length) {   // initialization of velocity
        //     velocity = new float[x_i.length];
        // }
        updateC1Schedule();         // we are updating c1 only for the neighborhood case
        Random rnd = new Random();
        
        logger.log("x_i.length = " +  x_i.length + ", count_updates: " + count_updates);

        for (int k = 0; k < x_i.length; k++) {

            float r1 = rnd.nextFloat();   // randomness
            float r2 = rnd.nextFloat();  

            inertiaVec[k] = W_INERTIA * velocity[k];
            cognitiveVec[k] = c1 * r1 * (pbest[k] - x_i[k]);
            // cognitiveVec[k] = C1 * r1 * (pbest[k] - x_i[k]);
            socialVec[k] = C2 * r2 * (gbest[k] - x_i[k]);
            diffPBestGBest[k] = C2 * r2 * (pbest[k] - gbest[k]);

            // float velocity_value = W_INERTIA * velocity[k] + C1 * r1 * (pbest[k] - x_i[k]) + C2 * r2 * (gbest[k] - x_i[k]);
            
            // float velocity_value = inertiaVec[k] + cognitiveVec[k] + socialVec[k];
            // velocity[k] = clampVelocity(velocity_value);  // velocity clamping implementation

            velocity[k] = inertiaVec[k] + cognitiveVec[k] + socialVec[k];

            // x_i_new[k] = x_i[k] + velocity[k];
        }

        clipVelocityByNorm(VMAX_NORM);

        for (int k = 0; k < x_i.length; k++) {
            x_i_new[k] = x_i[k] + velocity[k];
        }

        Dl4jParamUtils.updateModel(model, x_i_new);

        logger.log("PSO magnitudes: inertia = " + Dl4jParamUtils.magnitude(inertiaVec) + 
                ", cognitive = " + Dl4jParamUtils.magnitude(cognitiveVec) + ", with C1: " + c1 +
                ", social = " + Dl4jParamUtils.magnitude(socialVec) +
                ", diff = " + Dl4jParamUtils.magnitude(diffPBestGBest) + ", number of Clamps: " + clamp_count
        );

        iter++;

        return velocity;
    }

    //================================================================================================

    public float[] updateX(MultiLayerNetwork model, List<float[]> neighborPBestList) {      // for FULLY INFORMED

        count_updates++;

        Arrays.fill(socialVec, 0f);
        clamp_count = 0;

        float[] x_i = Dl4jParamUtils.modelToFlatList(model);
        Random rnd = new Random();

        logger.log("x_i.length = " +  x_i.length + ", count_updates: " + count_updates);

        // neighborPBestList empty case (initialization) ===================================================

        if (neighborPBestList == null || neighborPBestList.isEmpty()) {

            for (int k = 0; k < x_i.length; k++) {
                velocity[k] = W_INERTIA * velocity[k];
                x_i_new[k] = x_i[k] + velocity[k];
            }

            Dl4jParamUtils.updateModel(model, x_i_new);

            return this.velocity;
        }

        // Normal fully-informed case with non empty neighbors: =============================================

        for (float[] pBest_j : neighborPBestList) {

            if (pBest_j.length != x_i.length) {
                throw new IllegalArgumentException("pBest size mismatch");
            }

            for (int k = 0; k < x_i.length; k++) {
                float p_i_j = rnd.nextFloat();      // this is a uniformly distributed float value between 0.0 and 1.0
                socialVec[k] += p_i_j * (pBest_j[k] - x_i[k]);
            }
        }

        float scale = C / (float) N_WORKERS;
        for (int k = 0; k < socialVec.length; k++) {
            socialVec[k] *= scale;
            inertiaVec[k] = W_INERTIA * velocity[k];

            velocity[k] = inertiaVec[k] + socialVec[k];
            x_i_new[k] = x_i[k] + velocity[k];
        }

        logger.log("PSO magnitudes: inertia = " + Dl4jParamUtils.magnitude(inertiaVec) + ", social = " + Dl4jParamUtils.magnitude(socialVec) +
                    ", number of Clamps: " + clamp_count);
        
        // for (int k = 0; k < x_i.length; k++) {
        //     velocity[k] = clampVelocity(inertiaVec[k] + socialVec[k]);
        //     x_i_new[k] = x_i[k] + velocity[k];
        // }

        Dl4jParamUtils.updateModel(model, x_i_new);

        return this.velocity;
    }

    //================================================================================================

    // private void updateC1Schedule() {
        
    //     float t = Math.min(iter, MAX_ITERS);
    //     float alpha = t / (float) MAX_ITERS;          // 0 -> 1
    //     c1 = C1_START + alpha * (C1_END - C1_START);  // linearly moves start -> end
    // }

    // private void updateC1Schedule() {

    //     float t = Math.min(iter, MAX_ITERS) / (float) MAX_ITERS;  // [0,1]
    //     float k = 9.0f;   
    //     float sigmoid = (float)(1.0 / (1.0 + Math.exp(k * (t - 0.5))));

    //     c1 = C1_END + (C1_START - C1_END) * sigmoid;
    // }

    private void updateC1Schedule() {

        float u = Math.min(count_updates, C1_MAX_UPDATES);
        float k = (float)(2.0 * Math.log(9.0) / C1_DROP_WIDTH);
        float s = (float)(1.0 / (1.0 + Math.exp(k * (u - C1_MID_UPDATE))));

        c1 = C1_END + (C1_START - C1_END) * s;
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

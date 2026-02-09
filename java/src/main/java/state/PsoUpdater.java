package state;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.List;
import java.util.Random;
import java.util.Arrays;

import utils.*; 

public class PsoUpdater {

    private Config cfg = Config.getInstance();

    private final float W_INERTIA = cfg.W_INERTIA;
    private final float W_INERTIA_START = cfg.W_INERTIA;
    private final float W_INERTIA_END = 0.40f;
    private float W_INERTIA_CURRENT = cfg.W_INERTIA;

    private final float C = cfg.C;
    private final float C1 = cfg.C1;
    private final float C2 = cfg.C2;
    private final int N_WORKERS = cfg.N_WORKERS;
    private final int TRAIN_SIZE = cfg.TRAIN_SIZE;
    public final boolean SIMULATED_ANNEALING = cfg.SIMULATED_ANNEALING;

    private final float C1_START = cfg.C1; 
    private final float C1_END = 0.2f;
    private float c1 = C1_START;  
    private int iter = 0;
    private final int MAX_ITERS = 500;
    private final int NUM_SAMPLES = cfg.NUM_SAMPLES;
    private final int MAX_PSO_UPDATES = NUM_SAMPLES / (N_WORKERS * TRAIN_SIZE); // expected max updates (for clamping)
    private final int C1_MID_UPDATE = (int) Math.round(MAX_PSO_UPDATES / 1.6);
    private final float C1_DROP_WIDTH = 200f;   // the 200 means “mostly C1 drops between 600±100” → around 500–700

    private final float VMAX;    
    private final float VMAX_FACTOR = cfg.VMAX_FACTOR;;
    private final float VMAX_NORM;
    public final String VMAX_CLAMPING_TYPE = cfg.VMAX_CLAMPING_TYPE;

    private CustomLogger logger;
    private float[] velocity; 
    private float[] x_i_new;
    private float[] inertiaVec;
    private float[] cognitiveVec;
    private float[] socialVec;
    private float[] diffPBestGBest;

    private int clamp_count = 0;

    private int count_updates = 0;

    // Extra clamp parameters:
    private static final float W_MIN = 0.35f;     // exploitation
    private static final float W_MAX = 0.95f;     // exploration
    private static final float ACC_LOW  = 0.20f;  // below this: explore hard
    private static final float ACC_HIGH = 0.80f;  // above this: exploit
    private static final float EMA_ALPHA = 0.10f; // smoothing for noisy batch acc

    private float accEma = -1f;

    // Adaptive clamp
    private static final float VMAX_MIN = 0.005f;
    private static final float VMAX_MAX = 0.10f;

    //================================================================================================

    public PsoUpdater(MultiLayerNetwork model, int workerId) {
    
        float[] x = Dl4jParamUtils.modelToFlatList(model);
        x_i_new = new float[x.length];
        velocity = new float[x.length];
        inertiaVec = new float[x.length];
        cognitiveVec = new float[x.length];
        socialVec = new float[x.length];
        diffPBestGBest = new float[x.length];

        // float xmin = -1.0f; // Each individual weight is allowed to change at this rate
        // float xmax = 1.0f;  // During training, most weights should stay relatively small (in practice < 0.2 or < 0.5).
        // float range = xmax - xmin;  // the xmax - xmin, define the dynamic range. Dont enfoce xmax and xmin just use it to calculate dynamic range

        float range = computeDynamicRangeFromWeights(x);

        this.VMAX = this.VMAX_FACTOR * range;  // VMAX_FACTOR == the δ parameter (δ = VMAX_FACTOR)
        this.VMAX_NORM = (float)(Math.sqrt(x.length) * VMAX);

        randomizeVelocity(workerId, 0.1f); //  0.1f this affects the magnitude of the initialized velocity

        this.logger = CustomLogger.getWorkerInstance(workerId);

        logger.log("PsoUpdater: Number of weights (dimensionality): " + x.length + ", MAX_PSO_UPDATES: " + MAX_PSO_UPDATES + 
                ", C1_MID_UPDATE: " + C1_MID_UPDATE + "NUM_SAMPLES = " + NUM_SAMPLES);

    }

    //================================================================================================
    
    private float computeDynamicRangeFromWeights(float[] w) {
        // Percentile-like cheap approximation: use mean±3*std as "range"
        // (fast and no sorting)
        double mean = 0.0;
        for (float v : w) mean += v;
        mean /= w.length;

        double var = 0.0;
        for (float v : w) {
            double d = v - mean;
            var += d * d;
        }
        var /= w.length;
        double std = Math.sqrt(var);

        float k = 3.0f;
        float lo = (float)(mean - k * std);
        float hi = (float)(mean + k * std);
        return Math.max(1e-6f, hi - lo);
    }

    //================================================================================================

    private float clampVelocitySingle(float v) {      // this limits each coordinate Velocity independently

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

    private void clampVelocityByDim() {

        for (int i = 0; i < velocity.length; i++) {
            
            float v = velocity[i];
            if (v > VMAX) { 
                velocity[i] = VMAX; 
                clamp_count++; 
            } else if (v < -VMAX) { 
                velocity[i] = -VMAX; 
                clamp_count++; 
            }
        }
    }

    //================================================================================================

    private void clipVelocityByNorm(float vmaxNorm) {   // this is limiting overall length / magnitude of velocity
                                                        // the goal is to not limit individual dimensionalities, because this would change direction
        double sumSq = 0.0;
        for (float v : velocity) sumSq += (double)v * v;
        double norm = Math.sqrt(sumSq);

        if (norm > vmaxNorm && norm > 0.0) {
            float scale = (float)(vmaxNorm / norm);
            for (int i = 0; i < velocity.length; i++) velocity[i] *= scale;
        }
    }

    //================================================================================================
    // update for Neighborhood Best: 

    public float[] updateX(MultiLayerNetwork model, float[] pbest, float[] gbest) {     // FOR GBEST, not fully informed

        count_updates++;

        float[] x_i = Dl4jParamUtils.modelToFlatList(model);
        clamp_count = 0;

        updateParametersSchedule();         // we are updating c1 only for the neighborhood case
        Random rnd = new Random();
        
        logger.log("Count_updates: " + count_updates + ", Weight Dimensinality = " +  x_i.length);

        for (int k = 0; k < x_i.length; k++) {

            float r1 = rnd.nextFloat();   // randomness
            float r2 = rnd.nextFloat();  

            // inertiaVec[k] = W_INERTIA * velocity[k];
            inertiaVec[k] = W_INERTIA_CURRENT * velocity[k];

            if(SIMULATED_ANNEALING == false) {
                cognitiveVec[k] = C1 * r1 * (pbest[k] - x_i[k]);
            } else {
                cognitiveVec[k] = c1 * r1 * (pbest[k] - x_i[k]);
            }
            
            socialVec[k] = C2 * r2 * (gbest[k] - x_i[k]);
            diffPBestGBest[k] = C2 * r2 * (pbest[k] - gbest[k]);

            // float velocity_value = W_INERTIA * velocity[k] + C1 * r1 * (pbest[k] - x_i[k]) + C2 * r2 * (gbest[k] - x_i[k]);
            
            // float velocity_value = inertiaVec[k] + cognitiveVec[k] + socialVec[k];
            // velocity[k] = clampVelocity(velocity_value);  // velocity clamping implementation

            velocity[k] = inertiaVec[k] + cognitiveVec[k] + socialVec[k];

            // x_i_new[k] = x_i[k] + velocity[k];
        }

        if (VMAX_CLAMPING_TYPE.equals("DIM")) {
            clampVelocityByDim();     
        } else {
            clipVelocityByNorm(VMAX_NORM);
        }

        for (int k = 0; k < x_i.length; k++) {
            x_i_new[k] = x_i[k] + velocity[k];
        }

        Dl4jParamUtils.updateModel(model, x_i_new);

        logger.log("PSO magnitudes: inertia = " + Dl4jParamUtils.averageMagnitude(inertiaVec) + 
                ", with W_INERTIA: " + W_INERTIA_CURRENT +
                ", cognitive = " + Dl4jParamUtils.averageMagnitude(cognitiveVec) + 
                ", with C1: " + c1 +
                ", social = " + Dl4jParamUtils.averageMagnitude(socialVec) +
                ", diff = " + Dl4jParamUtils.averageMagnitude(diffPBestGBest) + 
                ", number of Clamps: " + clamp_count
        );

        iter++;

        return velocity;
    }

    // ================================================================================================
    // update for Fully Informed: 

    public float[] updateX(MultiLayerNetwork model, List<float[]> neighborPBestList) {      // for FULLY INFORMED

        count_updates++;

        Arrays.fill(socialVec, 0f);
        clamp_count = 0;

        float[] x_i = Dl4jParamUtils.modelToFlatList(model);
        Random rnd = new Random();

        logger.log("Count_updates: " + count_updates + ", Weight Dimensinality = " +  x_i.length);

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

        float scale = C / (float) neighborPBestList.size();

        // for (int k = 0; k < socialVec.length; k++) {
        //     socialVec[k] *= scale;
        //     inertiaVec[k] = W_INERTIA * velocity[k];

        //     velocity[k] = inertiaVec[k] + socialVec[k];
        //     x_i_new[k] = x_i[k] + velocity[k];
        // }

        for (int k = 0; k < socialVec.length; k++) {
            socialVec[k] *= scale;
            inertiaVec[k] = W_INERTIA * velocity[k];
            velocity[k] = inertiaVec[k] + socialVec[k];
        }

        if (VMAX_CLAMPING_TYPE.equals("DIM")) {
            clampVelocityByDim();     
        } else {
            clipVelocityByNorm(VMAX_NORM);
        }

        for (int k = 0; k < x_i.length; k++) {
            x_i_new[k] = x_i[k] + velocity[k];
        }

        logger.log("PSO magnitudes: inertia acc = " + Dl4jParamUtils.averageMagnitude(inertiaVec) + ", social = " + Dl4jParamUtils.averageMagnitude(socialVec) +
                    ", number of Clamps: " + clamp_count);
        
        // for (int k = 0; k < x_i.length; k++) {
        //     velocity[k] = clampVelocity(inertiaVec[k] + socialVec[k]);
        //     x_i_new[k] = x_i[k] + velocity[k];
        // }

        Dl4jParamUtils.updateModel(model, x_i_new);

        return this.velocity;
    }

    // ===============================================================================================================
    // ===============================================================================================================

    private float lerp(float a, float b, float t) {
        return a + (b - a) * t;
    }

    private float clamp01(float x) {    // be between 0 and 1
        return Math.max(0f, Math.min(1f, x));
    }

    /**
     * Maps accuracy -> inertia:
     *   acc <= ACC_LOW  => W_MAX
     *   acc >= ACC_HIGH => W_MIN
     * linear in between
     */
    private float adaptiveInertia(float acc) {
        // normalize acc into [0..1] within [ACC_LOW..ACC_HIGH]
        float t = (acc - ACC_LOW) / (ACC_HIGH - ACC_LOW);
        t = clamp01(t);

        // t=0 => low acc => W_MAX
        // t=1 => high acc => W_MIN
        return lerp(W_MAX, W_MIN, t);
    }

    // public float[] updateXAdaptive(
    //         MultiLayerNetwork model,
    //         List<float[]> neighborPBestList,
    //         float batchAccuracy
    // ) {
    //     count_updates++;

    //     Arrays.fill(socialVec, 0f);
    //     clamp_count = 0;

    //     float[] x_i = Dl4jParamUtils.modelToFlatList(model);

    //     // Smooth accuracy (important because batch accuracy is noisy)
    //     if (accEma < 0f) accEma = batchAccuracy;
    //     accEma = (1f - EMA_ALPHA) * accEma + EMA_ALPHA * batchAccuracy;

    //     // Compute adaptive parameters
    //     float w = adaptiveInertia(accEma);

    //     Random rnd = new Random();

    //     logger.log("Count_updates=" + count_updates
    //             + " acc=" + batchAccuracy
    //             + " accEma=" + accEma
    //             + " w=" + w
    //             + " dim=" + x_i.length);

    //     // Initialization case (no neighbors yet)
    //     if (neighborPBestList == null || neighborPBestList.isEmpty()) {
    //         for (int k = 0; k < x_i.length; k++) {
    //             x_i_new[k] = x_i[k] + w * velocity[k];
    //         }
    //         Dl4jParamUtils.updateModel(model, x_i_new);
    //         return this.velocity;
    //     }

    //     // Social term
    //     for (float[] pBest_j : neighborPBestList) {
    //         if (pBest_j.length != x_i.length) {
    //             throw new IllegalArgumentException("pBest size mismatch");
    //         }
    //         for (int k = 0; k < x_i.length; k++) {
    //             float r = rnd.nextFloat();
    //             socialVec[k] += r * (pBest_j[k] - x_i[k]);
    //         }
    //     }

    //     float scale = C / (float) neighborPBestList.size();

    //     for (int k = 0; k < x_i.length; k++) {
    //         socialVec[k] *= scale;
    //         inertiaVec[k] = w * velocity[k];

    //         velocity[k] = inertiaVec[k] + socialVec[k];
    //         x_i_new[k] = x_i[k] + velocity[k];
    //     }

    //     logger.log("magnitudes: inertia=" + Dl4jParamUtils.averageMagnitude(inertiaVec)
    //             + " social=" + Dl4jParamUtils.averageMagnitude(socialVec)
    //             + " clamps=" + clamp_count);

    //     Dl4jParamUtils.updateModel(model, x_i_new);
    //     return this.velocity;
    // }


    //================================================================================================

    // private void updateParametersSchedule() {
        
    //     float t = Math.min(iter, MAX_ITERS);
    //     float alpha = t / (float) MAX_ITERS;          // 0 -> 1
    //     c1 = C1_START + alpha * (C1_END - C1_START);  // linearly moves start -> end
    // }

    // private void updateParametersSchedule() {

    //     float t = Math.min(iter, MAX_ITERS) / (float) MAX_ITERS;  // [0,1]
    //     float k = 9.0f;   
    //     float sigmoid = (float)(1.0 / (1.0 + Math.exp(k * (t - 0.5))));

    //     c1 = C1_END + (C1_START - C1_END) * sigmoid;
    // }

    private void updateParametersSchedule() {

        float u = Math.min(count_updates, MAX_PSO_UPDATES); // makes u not surpass MAX_PSO_UPDATES
        float k = (float)(2.0 * Math.log(9.0) / C1_DROP_WIDTH);
        float s = (float)(1.0 / (1.0 + Math.exp(k * (u - C1_MID_UPDATE))));

        c1 = C1_END + (C1_START - C1_END) * s;

        float t = u / (float) MAX_PSO_UPDATES;   
        W_INERTIA_CURRENT = W_INERTIA_START + t * (W_INERTIA_END - W_INERTIA_START);  // t = [0, 1]
            // when t = 1, then W_INERTIA_CURRENT == W_INERTIA_END. This is linear fall

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

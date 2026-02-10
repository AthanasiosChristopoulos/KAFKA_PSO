package state;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.List;
import java.util.Random;
import java.util.Arrays;

import utils.*; 

public class PsoUpdater {

    private Config cfg = Config.getInstance();

    private final float W_INERTIA = cfg.W_INERTIA;
    private final float W_INERTIA_START = cfg.W_INERTIA_START;
    private final float W_INERTIA_END = cfg.W_INERTIA_END;
    private float W_INERTIA_CURRENT = cfg.W_INERTIA;
    public final boolean ADAPTIVE_INERTIA = cfg.ADAPTIVE_INERTIA;

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
    private final float VMAX_FACTOR = cfg.VMAX_FACTOR;
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

    private final Random rnd;

    // ---- Adaptive inertia (progress-based) ----
    private float wCurrent = cfg.W_INERTIA_START;   // start high
    private float bestAccEma = -1f;                 // best (so far) EMA of accuracy
    private float accEma = -1f;                     // EMA of current batch accuracy

    private final float W_MIN_ADAPT = cfg.W_INERTIA_END; // exploit
    private final float W_MAX_ADAPT = cfg.W_INERTIA_START; // explore
    private final float ACC_EMA_ALPHA = 0.10f;  // smoothing
    private final float IMPROVE_EPS = 0.002f;    // “no progress” threshold (0.2% acc)
    private final float W_STEP_UP = 0.02f;       // explore increase step
    private final float W_STEP_DOWN = 0.01f;     // exploit decrease step

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
                ", C1_MID_UPDATE: " + C1_MID_UPDATE + ", NUM_SAMPLES = " + NUM_SAMPLES);

        this.rnd = new Random(1234L + workerId);    // for extra randomness in between workers
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

    private float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
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

    public float[] updateX(MultiLayerNetwork model, float[] x_i, float[] pbest, float[] gbest, float batchAccuracy, String taskInstance) {     // FOR GBEST, not fully informed

        count_updates++;

        clamp_count = 0;

        if(ADAPTIVE_INERTIA) {
            updateParametersSchedule(); 
            // updateInertiaFromProgress(batchAccuracy);   // adaptive inertia   
        }
        
        // logger.log("Count_updates: " + count_updates);

        // float r1 = rnd.nextFloat();   // randomness. Is not dimensional, it is a factor equal in all dimensions
        // float r2 = rnd.nextFloat();  

        for (int k = 0; k < x_i.length; k++) {

            float r1 = rnd.nextFloat();   // randomness. Is dimensional, for every other dimension this is randomly changed
            float r2 = rnd.nextFloat();  

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
            x_i[k] = x_i[k] + velocity[k];
        }

        Dl4jParamUtils.updateModel(model, x_i);

        logger.log(taskInstance + ", PSO magnitudes: " + 
                "inertia = " + Dl4jParamUtils.rmsScaled(inertiaVec, 100) + 
                ", with W_INERTIA: " + W_INERTIA_CURRENT +
                ", cognitive = " + Dl4jParamUtils.rmsScaled(cognitiveVec, 100) + 
                ", social = " + Dl4jParamUtils.rmsScaled(socialVec, 100) +
                ", diffPBestGBest = " + Dl4jParamUtils.rmsScaled(diffPBestGBest, 100) + 
                ", number of Clamps: " + clamp_count +
                ", count_updates: " + count_updates
        );

        iter++;

        return velocity;
    }

    // ================================================================================================
    // update for Fully Informed: 

    public float[] updateX(MultiLayerNetwork model, float[] x_i, List<float[]> neighborPBestList, float batchAccuracy, String taskInstance) {      // for FULLY INFORMED

        count_updates++;

        Arrays.fill(socialVec, 0f);
        clamp_count = 0;

        if(ADAPTIVE_INERTIA) {
            updateParametersSchedule();
            // updateInertiaFromProgress(batchAccuracy);   // adaptive inertia   
        }

        // logger.log("Count_updates: " + count_updates);

        // neighborPBestList empty case (initialization) ===================================================

        if (neighborPBestList == null || neighborPBestList.isEmpty()) {

            for (int k = 0; k < x_i.length; k++) {
                velocity[k] = W_INERTIA_CURRENT * velocity[k];
                x_i[k] = x_i[k] + velocity[k];
            }
            Dl4jParamUtils.updateModel(model, x_i);
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

        for (int k = 0; k < socialVec.length; k++) {
            socialVec[k] *= scale;
            inertiaVec[k] = W_INERTIA_CURRENT * velocity[k];
            velocity[k] = inertiaVec[k] + socialVec[k];
        }

        if (VMAX_CLAMPING_TYPE.equals("DIM")) {
            clampVelocityByDim();     
        } else {
            clipVelocityByNorm(VMAX_NORM);
        }

        for (int k = 0; k < x_i.length; k++) {
            x_i[k] = x_i[k] + velocity[k];
        }
        Dl4jParamUtils.updateModel(model, x_i);
        logger.log(taskInstance + ", PSO magnitudes: " +
                    "inertia = " + Dl4jParamUtils.rmsScaled(inertiaVec, 100) + 
                    ", with W_INERTIA: " + W_INERTIA_CURRENT +
                    ", social = " + Dl4jParamUtils.rmsScaled(socialVec, 100) +
                    ", number of Clamps: " + clamp_count + 
                    ", count_updates: " + count_updates
        );

        return this.velocity;
    }

    // ===============================================================================================================
    // updateParameters during the run:

    private void updateParametersSchedule() {

        float updateIndex = Math.min(count_updates, MAX_PSO_UPDATES); // makes updateIndex not surpass MAX_PSO_UPDATES

        float sigmoidSteepness = (float)(2.0 * Math.log(9.0) / C1_DROP_WIDTH);
        float retentionFactor = (float)(1.0 / (1.0 + Math.exp(sigmoidSteepness * (updateIndex - C1_MID_UPDATE))));
        c1 = C1_END + (C1_START - C1_END) * retentionFactor;
            // this is exponential fall, right around the middle

        float progressFactor = updateIndex / (float) MAX_PSO_UPDATES;   
        W_INERTIA_CURRENT = W_INERTIA_START + progressFactor * (W_INERTIA_END - W_INERTIA_START);  // t = [0, 1]
            // when t = 1, then W_INERTIA_CURRENT == W_INERTIA_END. This is linear fall of INERTIA
    }

    //================================================================================================

    private void updateInertiaFromProgress(float batchAccuracy) {

        if (accEma < 0f) accEma = batchAccuracy;  // Initialization
    
        accEma = (1f - ACC_EMA_ALPHA) * accEma + ACC_EMA_ALPHA * batchAccuracy; // EMA = Exponential Moving Average
            // EMA == What is the recent trend of accuracy, not just this one batch ?

        if (bestAccEma < 0f) bestAccEma = accEma;   // init best

        boolean improved = accEma > bestAccEma + IMPROVE_EPS;

        if (improved) {
            bestAccEma = accEma;
            wCurrent -= W_STEP_DOWN;   // exploit more
        } else {
            wCurrent += W_STEP_UP;     // explore more
        }

        wCurrent = clamp(wCurrent, W_MIN_ADAPT, W_MAX_ADAPT);

        W_INERTIA_CURRENT = wCurrent;  // keep your existing variable as the "source of truth"
    }

    //================================================================================================

    public void randomizeVelocity(int workerId, float sigma) {
        Random random = new Random(workerId);

        for (int i = 0; i < velocity.length; i++) {
            velocity[i] = (float) random.nextGaussian() * sigma;
        }
    }

    //================================================================================================
    // Random Weight Model Initialization  
    // unnecessary since model.init(); with .seed(123) does this deterministic weight initialization => good for debugging
    // public void randomizeModelWeights(MultiLayerNetwork model, int seed, float sigma) {

    //     float[] flat = Dl4jParamUtils.modelToFlatList(model);
    //     Random random = new Random(seed);

    //     for (int i = 0; i < flat.length; i++) {
    //         flat[i] += random.nextGaussian() * sigma;
    //     }

    //     Dl4jParamUtils.updateModel(model, flat);
    // }

}

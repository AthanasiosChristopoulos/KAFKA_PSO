package state;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.List;
import java.util.Random;
import java.util.Arrays;

import utils.*; 

public class PsoUpdater {

    private final float W_INERTIA;
    private final float C;
    private final float C1;
    private final float C2;
    private final int N_WORKERS;

    private final float VMAX;    
    private final float VMAX_FACTOR;

    private CustomLogger logger;
    private float[] velocity; 
    private float[] x_i_new;
    private float[] inertiaVec;
    private float[] cognitiveVec;
    private float[] socialVec;

    private int clamp_count = 0;

    public PsoUpdater(MultiLayerNetwork model, int workerId) {
        
        Config cfg = Config.getInstance();

        this.W_INERTIA = cfg.W_INERTIA;
        this.C = cfg.C;
        this.C1 = cfg.C1;
        this.C2 = cfg.C2;

        this.N_WORKERS = cfg.N_WORKERS;

        float[] x = Dl4jParamUtils.modelToFlatList(model);
        x_i_new = new float[x.length];
        velocity = new float[x.length];
        inertiaVec = new float[x.length];
        cognitiveVec = new float[x.length];
        socialVec = new float[x.length];

        this.VMAX_FACTOR = cfg.VMAX_FACTOR;

        float xmin = -1.0f; // Each individual weight is allowed to change at this rate
        float xmax = 1.0f;  // During training, most weights should stay relatively small (in practice < 0.2 or < 0.5).

        float range = xmax - xmin;  // the xmax - xmin discussed in the paper 

        this.VMAX = VMAX_FACTOR * range;  // VMAX_FACTOR == the δ discussed in the paper 

        randomizeVelocity(workerId, 0.1f); //  0.1f this affects the magnitude of the initialized velocity

        this.logger = CustomLogger.getWorkerInstance(workerId);
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

    public float[] updateX(MultiLayerNetwork model, float[] pbest, float[] gbest) {     // FOR GBEST, not fully informed

        float[] x_i = Dl4jParamUtils.modelToFlatList(model);
        clamp_count = 0;
        // if (velocity == null || velocity.length != x_i.length) {   // initialization of velocity
        //     velocity = new float[x_i.length];
        // }

        Random rnd = new Random();

        for (int k = 0; k < x_i.length; k++) {

            float r1 = rnd.nextFloat();   // randomness
            float r2 = rnd.nextFloat();  

            inertiaVec[k] = W_INERTIA * velocity[k];
            cognitiveVec[k] = C1 * r1 * (pbest[k] - x_i[k]);
            socialVec[k] = C2 * r2 * (gbest[k] - x_i[k]);

            // float velocity_value = W_INERTIA * velocity[k] + C1 * r1 * (pbest[k] - x_i[k]) + C2 * r2 * (gbest[k] - x_i[k]);
            
            float velocity_value = inertiaVec[k] + cognitiveVec[k] + socialVec[k];

            // velocity_i_1[k] = inertia + cognitive + social;
            velocity[k] = clampVelocity(velocity_value);  // velocity clamping implementation
            // velocity_i_1[k] = velocity_value;
            x_i_new[k] = x_i[k] + velocity[k];
        }

        Dl4jParamUtils.updateModel(model, x_i_new);

        logger.log("PSO magnitudes: inertia = " + Dl4jParamUtils.magnitude(inertiaVec) + 
                ", cognitive = " + Dl4jParamUtils.magnitude(cognitiveVec) + ", social = " + Dl4jParamUtils.magnitude(socialVec) +
                ", number of Clamps: " + clamp_count
        );

        return velocity;
    }

    //================================================================================================

    public float[] updateX(MultiLayerNetwork model, List<float[]> neighborPBestList) {      // for FULLY INFORMED
        
        Arrays.fill(socialVec, 0f);
        clamp_count = 0;

        float[] x_i = Dl4jParamUtils.modelToFlatList(model);
        Random rnd = new Random();

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
                float p_i_j = rnd.nextFloat();  
                socialVec[k] += p_i_j * (pBest_j[k] - x_i[k]);
            }
        }

        float scale = C / (float) N_WORKERS;
        for (int k = 0; k < socialVec.length; k++) {
            socialVec[k] *= scale;
            inertiaVec[k] = W_INERTIA * velocity[k];
        }

        logger.log("PSO magnitudes: inertia = " + Dl4jParamUtils.magnitude(inertiaVec) + ", social = " + Dl4jParamUtils.magnitude(socialVec));
        
        for (int k = 0; k < x_i.length; k++) {
            velocity[k] = clampVelocity(inertiaVec[k] + socialVec[k]);
            x_i_new[k] = x_i[k] + velocity[k];
        }

        Dl4jParamUtils.updateModel(model, x_i_new);

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

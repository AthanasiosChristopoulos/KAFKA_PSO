package state;

import dl4j_models.Dl4jParamUtils;

import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.Arrays;

import utils.*; 

public class PsoUpdater {

    private Config cfg = Config.getInstance();
    private final float INERTIA_START = cfg.INERTIA_START;
    private final float INERTIA_END = cfg.INERTIA_END;
    private float INERTIA_CURRENT = cfg.INERTIA;
    public final boolean ADAPTIVE_INERTIA = cfg.ADAPTIVE_INERTIA;

    private final float C = cfg.C;
    private final float C1 = cfg.C1;
    private final float C2 = cfg.C2;
    private final int BATCH_SIZE = cfg.BATCH_SIZE;
    public final boolean INCLUDE_SELF = cfg.INCLUDE_SELF;
    public final boolean GIVE_HALF_TO_SELF = cfg.GIVE_HALF_TO_SELF;
    public final boolean WEIGHTS_ON_UPDATEX = cfg.WEIGHTS_ON_UPDATEX;

    public final boolean ACCELARATION_COEFF_TIME_VARYING = cfg.ACCELARATION_COEFF_TIME_VARYING;
    private float C1_START = cfg.C1_START; 
    private float C1_END = cfg.C1_END;
    private float C2_START = cfg.C2_START; 
    private float C2_END = cfg.C2_END;
    private float C_START = cfg.C_START; 
    private float C_END = cfg.C_END;
    
    private float c1;  
    private float c2;  
    private float c;  

    private final int NUM_SAMPLES = cfg.NUM_SAMPLES;
    private final int MAX_UPDATES; 

    private final int C1_MID_UPDATE;

    private final float VMAX;    
    private final float VMAX_FACTOR = cfg.VMAX_FACTOR;
    private final float VMAX_NORM;
    public final String VMAX_CLAMPING_TYPE = cfg.VMAX_CLAMPING_TYPE;

    private final float WEIGHTS_INIT_SCALE = cfg.WEIGHTS_INIT_SCALE;
    public final boolean WEIGHT_CLAMPING = cfg.WEIGHT_CLAMPING;
    public final float WEIGHT_MAX_SCALE = cfg.WEIGHT_MAX_SCALE; 

    private CustomLogger logger;
    private float[] velocity; 
    private float[] inertiaVec;
    private float[] cognitiveVec;
    private float[] socialVec;
    private float[] diffPBestGBest;

    private int clamp_count = 0;

    private int count_updates = 0;

    private final SplittableRandom rnd;     // helps performance in comparison to a normal randomizer

    private final WorkerStatic ws;
    private final int dimensionality;

    private float[] num;
    private float[] den;

    private float EPS = 0.0000001f; 

    private double velocity_norm;

    //================================================================================================

    public PsoUpdater(int workerId, WorkerStatic ws) {

        dimensionality = ws.flatModel.length;
        velocity = new float[dimensionality];
        inertiaVec = new float[dimensionality];
        cognitiveVec = new float[dimensionality];
        socialVec = new float[dimensionality];
        diffPBestGBest = new float[dimensionality];

        this.ws = ws;

        float range = computeDynamicRangeFromWeights(ws.flatModel);
        this.VMAX = this.VMAX_FACTOR * range;  // VMAX_FACTOR == the δ parameter (δ = VMAX_FACTOR)
        this.VMAX_NORM = (float)(Math.sqrt(dimensionality) * VMAX);

        randomizeVelocity(workerId, WEIGHTS_INIT_SCALE * 0.1f); // 0.1f this affects the magnitude of the initialized velocity. velocity is initialized uniquelly / seeded by workerId

        this.logger = CustomLogger.getWorkerInstance(workerId);

        if(!cfg.DATASET_PARTITIONING) {
            MAX_UPDATES = NUM_SAMPLES / (3 * BATCH_SIZE);   // (* 3): This is necessary because othewise it will never converge. 
            
        } else {
            MAX_UPDATES = NUM_SAMPLES / (cfg.N_WORKERS * BATCH_SIZE);
        }
        C1_MID_UPDATE = (int) Math.round(MAX_UPDATES / 1.6);

        if (logger.isEnabled(2)) logger.log("PsoUpdater: Number of weights (dimensionality): " + 
                dimensionality + ", MAX_UPDATES: " + MAX_UPDATES + 
                ", C1_MID_UPDATE: " + C1_MID_UPDATE + ", NUM_SAMPLES = " + NUM_SAMPLES);

        this.rnd = new SplittableRandom(1234L + workerId);      // more efficient than simple random

        this.num = new float[dimensionality];
        this.den = new float[dimensionality];

        if(ACCELARATION_COEFF_TIME_VARYING) {
            this.c1 = C1_START;
            this.c2 = C2_START;
            this.c = C_START;
        } else {
            this.c1 = C1;
            this.c2 = C2;
            this.c = C;
        }             
    }

    //================================================================================================
    
    private float computeDynamicRangeFromWeights(float[] w) {
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

    private void clampVelocityByDim() {

        for (int i = 0; i < dimensionality; i++) {
            
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

    private void clipVelocityByNorm() {   // this is limiting overall length / magnitude of velocity
                                        // the goal is to not limit individual dimensionalities, because this would change direction
        double sumSq = 0.0;
        for (float v : velocity) sumSq += (double)v * v;
        velocity_norm = Math.sqrt(sumSq);

        if (velocity_norm > VMAX_NORM) {

            float scale = (float)(VMAX_NORM / velocity_norm);   // this means that the scale (hollistically for all dimensionalities), are going to be VMAX_NORM. Essentially we are doing: velocity_norm * (VMAX_NORM / velocity_norm)

            for (int i = 0; i < dimensionality; i++) {
                velocity[i] *= scale;
                clamp_count++; 
            }
        }
    }

    //================================================================================================
    // update for Neighborhood Best (GBEST): 

    public float[] updateX(float[] pbest, float[] gbest, float batchAccuracy, String taskInstance) {     

        count_updates++;

        clamp_count = 0;

        if(ADAPTIVE_INERTIA) updateParametersSchedule(); 
        if(ACCELARATION_COEFF_TIME_VARYING) updateC1C2Schedule();

        for (int k = 0; k < dimensionality; k++) {

            float r1 = rnd.nextFloat();   // randomness. For every other dimension this is randomly changed
            float r2 = rnd.nextFloat();  

            inertiaVec[k] = INERTIA_CURRENT * velocity[k];

            cognitiveVec[k] = c1 * r1 * (pbest[k] - ws.flatModel[k]);
            socialVec[k] = c2 * r2 * (gbest[k] - ws.flatModel[k]);
            diffPBestGBest[k] = c2 * r2 * (pbest[k] - gbest[k]);

            velocity[k] = inertiaVec[k] + cognitiveVec[k] + socialVec[k];
        }

        if (VMAX_CLAMPING_TYPE.equals("DIM")) {
            clampVelocityByDim();     
        } else {
            clipVelocityByNorm();
        }

        if (WEIGHT_CLAMPING) {
            for (int k = 0; k < dimensionality; k++) {
                ws.flatModel[k] = clamp(ws.flatModel[k] + velocity[k], - WEIGHT_MAX_SCALE, WEIGHT_MAX_SCALE);
            }
        } else {
            for (int k = 0; k < dimensionality; k++) {
                ws.flatModel[k] = ws.flatModel[k] + velocity[k];
            }
        }
        
        if(cfg.USING_PRETRAINED_MODEL) {
            Dl4jParamUtils.updateModelHead(ws.model, ws.flatModel, ws.start);
        } else {
            Dl4jParamUtils.updateModel(ws.model, ws.flatModel);
        }
        
        if(VMAX_CLAMPING_TYPE.equals("NORM")) {
            if (logger.isEnabled(0)) logger.log(taskInstance + ", PSO magnitudes: " + 
                    "inertia = " + Dl4jParamUtils.rmsScaled(inertiaVec, 100) + 
                    ", with INERTIA: " + INERTIA_CURRENT +
                    ", cognitive = " + Dl4jParamUtils.rmsScaled(cognitiveVec, 100) + ", c1: " + Dl4jParamUtils.round(c1, 1) +
                    ", social = " + Dl4jParamUtils.rmsScaled(socialVec, 100) + ", c2: " + Dl4jParamUtils.round(c2, 1) +
                    ", diffPBestGBest = " + Dl4jParamUtils.rmsScaled(diffPBestGBest, 100) + 
                    ", VMAX_NORM: " + VMAX_NORM + ", velocity_norm: " + velocity_norm + 
                    ", count_updates: " + count_updates
            );
        } else {
        
            if (logger.isEnabled(0)) logger.log(taskInstance + ", PSO magnitudes: " + 
                    "inertia = " + Dl4jParamUtils.rmsScaled(inertiaVec, 100) + 
                    ", with INERTIA: " + INERTIA_CURRENT +
                    ", cognitive = " + Dl4jParamUtils.rmsScaled(cognitiveVec, 100) + ", c1: " + Dl4jParamUtils.round(c1, 1) +
                    ", social = " + Dl4jParamUtils.rmsScaled(socialVec, 100) + ", c2: " + Dl4jParamUtils.round(c2, 1) +
                    ", diffPBestGBest = " + Dl4jParamUtils.rmsScaled(diffPBestGBest, 100) + 
                    ", number of Clamps: " + clamp_count +
                    ", count_updates: " + count_updates
            );

        }

        return velocity;
    }

    // ================================================================================================
    // update for Fully Informed: 

    public float[] updateX(List<NeighborPBest> neighbors, float batchAccuracy, String taskInstance) {     

        count_updates++;
        clamp_count = 0;

        if(ADAPTIVE_INERTIA) updateParametersSchedule(); 
        if(ACCELARATION_COEFF_TIME_VARYING) updateC1C2Schedule();

        // neighbors.pBest empty case (initialization) ===================================================

        if (neighbors == null || neighbors.isEmpty()) {

            for (int k = 0; k < dimensionality; k++) {
                velocity[k] = INERTIA_CURRENT * velocity[k];
            }

            if (WEIGHT_CLAMPING) {
                for (int k = 0; k < dimensionality; k++) {
                    ws.flatModel[k] = clamp(ws.flatModel[k] + velocity[k], - WEIGHT_MAX_SCALE, WEIGHT_MAX_SCALE);
                }
            } else {
                for (int k = 0; k < dimensionality; k++) {
                    ws.flatModel[k] = ws.flatModel[k] + velocity[k];
                }
            }

            if(cfg.USING_PRETRAINED_MODEL) {
                Dl4jParamUtils.updateModelHead(ws.model, ws.flatModel, ws.start);
            } else {
                Dl4jParamUtils.updateModel(ws.model, ws.flatModel);
            }

            return this.velocity;
        }

        int N = neighbors.size();
        Arrays.fill(num, 0f);
        Arrays.fill(den, 0f);

        final float phiMax;
        
        if(GIVE_HALF_TO_SELF) {
            phiMax = 0.5f * c; 
            if(INCLUDE_SELF && N > 1) {
                N = neighbors.size() - 1;
            }
        } else {
            phiMax = c; 
        }
            
        final float phiMaxPerNeighbor = phiMax / (float) N;
        float accWk = 0;

        if(WEIGHTS_ON_UPDATEX) {

            for (NeighborPBest nb : neighbors) {
                if(GIVE_HALF_TO_SELF && ws.workerId == nb.workerId) {
                    continue;
                }
                float Wk = Math.max(EPS, nb.accuracy); 
                accWk += Wk;

                float[] Pk = nb.pBest;

                for (int d = 0; d < dimensionality; d++) {

                    float phi_kd = (float) rnd.nextDouble() * phiMaxPerNeighbor; // U[0, C/N]
                    
                    float wphi = Wk * phi_kd;

                    num[d] += wphi * Pk[d];
                    den[d] += wphi;
                }
            }
            float scaleWk = accWk / N;

            for (int d = 0; d < dimensionality; d++) {
                float Pm_d = (den[d] > EPS) ? (num[d] / den[d]) : ws.flatModel[d];

                inertiaVec[d] = INERTIA_CURRENT * velocity[d];
                socialVec[d]  = (den[d] / scaleWk) * (Pm_d - ws.flatModel[d]); 

                if(GIVE_HALF_TO_SELF) {
                    cognitiveVec[d] =  phiMax * rnd.nextFloat() * (ws.pBestWeights[d] - ws.flatModel[d]);
                    velocity[d] = inertiaVec[d] + cognitiveVec[d] + socialVec[d];
                } else {
                    velocity[d] = inertiaVec[d] + socialVec[d]; 
                }
            }

        } else {

            for (NeighborPBest nb : neighbors) {
                if(GIVE_HALF_TO_SELF && ws.workerId == nb.workerId) {
                    continue;
                }

                float[] Pk = nb.pBest;

                for (int d = 0; d < dimensionality; d++) {

                    float phi_kd = (float) rnd.nextDouble() * phiMaxPerNeighbor; // U[0, C/N]
                    num[d] += phi_kd * Pk[d];
                    den[d] += phi_kd;
                }
            }

            for (int d = 0; d < dimensionality; d++) {
                float Pm_d = (den[d] > EPS) ? (num[d] / den[d]) : ws.flatModel[d];

                inertiaVec[d] = INERTIA_CURRENT * velocity[d];

                socialVec[d]  = den[d] * (Pm_d - ws.flatModel[d]);  

                if(GIVE_HALF_TO_SELF) {
                    cognitiveVec[d] =  phiMax * rnd.nextFloat() * (ws.pBestWeights[d] - ws.flatModel[d]);
                    velocity[d] = inertiaVec[d] + cognitiveVec[d] + socialVec[d];
                } else {
                    velocity[d] = inertiaVec[d] + socialVec[d];
                }
            
            }
        }

        // ===============================================================================

        if (VMAX_CLAMPING_TYPE.equals("DIM")) {
            clampVelocityByDim();     
        } else {
            clipVelocityByNorm();
        }

        if (WEIGHT_CLAMPING) {
            for (int k = 0; k < dimensionality; k++) {
                ws.flatModel[k] = clamp(ws.flatModel[k] + velocity[k], - WEIGHT_MAX_SCALE, WEIGHT_MAX_SCALE);
            }
        } else {
            for (int k = 0; k < dimensionality; k++) {
                ws.flatModel[k] = ws.flatModel[k] + velocity[k];
            }
        }
                
        if(cfg.USING_PRETRAINED_MODEL) {
            Dl4jParamUtils.updateModelHead(ws.model, ws.flatModel, ws.start);
        } else {
            Dl4jParamUtils.updateModel(ws.model, ws.flatModel);
        }
        
            
        if(VMAX_CLAMPING_TYPE.equals("NORM")) {
            if (logger.isEnabled(0)) logger.log(taskInstance + ", PSO magnitudes: " +
                        "inertia = " + Dl4jParamUtils.rmsScaled(inertiaVec, 100) + 
                        ", with INERTIA: " + INERTIA_CURRENT +
                        ", social = " + Dl4jParamUtils.rmsScaled(socialVec, 100) +
                        ", VMAX_NORM: " + VMAX_NORM + ", velocity_norm: " + velocity_norm + 
                        ", count_updates: " + count_updates
            );
        } else {
            if (logger.isEnabled(0)) logger.log(taskInstance + ", PSO magnitudes: " +
                        "inertia = " + Dl4jParamUtils.rmsScaled(inertiaVec, 100) + 
                        ", with INERTIA: " + INERTIA_CURRENT +
                        ", social = " + Dl4jParamUtils.rmsScaled(socialVec, 100) +
                        ", number of Clamps: " + clamp_count + 
                        ", count_updates: " + count_updates
            );
        }

        return this.velocity;
    }

    // ===============================================================================================================
    // updateParameters during the run:

    private void updateParametersSchedule() {

        float updateIndex = Math.min(count_updates, MAX_UPDATES); // makes updateIndex not surpass MAX_UPDATES

        float progressFactor = updateIndex / (float) MAX_UPDATES;   
        INERTIA_CURRENT = INERTIA_START + progressFactor * (INERTIA_END - INERTIA_START);  // t = [0, 1]
            // when t = 1, then INERTIA_CURRENT == INERTIA_END. This is linear fall of INERTIA
        // Behavior explained: 
        // if progressFactor = 0 (if updateIndex == MAX_UPDATES), then INERTIA_CURRENT = INERTIA_MIN
    }

    //================================================================================================

    private void updateC1C2Schedule() {

        float updateIndex = Math.min(count_updates, MAX_UPDATES); // makes updateIndex not surpass MAX_UPDATES
        float progressFactor = updateIndex / (float) MAX_UPDATES;  
        
        c1 = (C1_END - C1_START) * progressFactor + C1_START;
        c2 = (C2_END - C2_START) * progressFactor + C2_START;
        c =  (C_END - C_START) * progressFactor + C_START;
    }

    //================================================================================================

    public void randomizeVelocity(int workerId, float sigma) {

        Random random = new Random(workerId);

        for (int i = 0; i < dimensionality; i++) {
            velocity[i] = (float) random.nextGaussian() * sigma;
        }
    }
}

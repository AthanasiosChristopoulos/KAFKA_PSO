package utils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import state.*;

public final class WorkerStatic {

    // One entry per workerId
    private static final ConcurrentHashMap<Integer, WorkerStatic> INSTANCES = new ConcurrentHashMap<>();

    public static WorkerStatic get(int workerId) {
        return INSTANCES.computeIfAbsent(workerId, WorkerStatic::new);
    }

    // ===================== SHARED STATE =====================

    public final int workerId;

    public final MultiLayerNetwork model;
    public float[] pBestWeights;

    public final Stats stats;
    public final PsoUpdater psoUpdater;
    public final BatchPrediction predictor;

    public float local_gBestLoss = 100000f;
    public float local_gBestAccuracy = -1f;

    public int batchesRead = 0;
    
    // ========================================================

    private WorkerStatic(int workerId) {
        
        this.workerId = workerId;
        this.model = Dl4jModelFactory.createModel();
        this.stats = new Stats();
        this.psoUpdater = new PsoUpdater(model, workerId);
        this.predictor = new BatchPrediction(model, CustomLogger.getWorkerInstance(workerId)); 
        this.pBestWeights = Dl4jParamUtils.modelToFlatList(model);
    }
}

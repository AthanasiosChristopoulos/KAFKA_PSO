package utils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import state.*;

public final class WorkerStatic {

    private static Config cfg = Config.getInstance();
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;

    private static final ConcurrentHashMap<Integer, WorkerStatic> INSTANCES = new ConcurrentHashMap<>();

    public static WorkerStatic get(int workerId) {
        return INSTANCES.computeIfAbsent(workerId, WorkerStatic::new);
    }

    // ===================== SHARED STATE =====================

    public final int workerId;

    public final MultiLayerNetwork model;
    public float[] flatModel;

    public float[] pBestWeights;

    public final Stats stats;
    public final PsoUpdater psoUpdater;
    public final BatchPrediction predictor;

    private CustomLogger logger;

    public float local_gBestLoss = 100000f;
    public float local_gBestAccuracy = -1f;

    public int batchesRead = 0;
    public boolean printedReport = false;
    public boolean endedWorker = false;

    public int inactivePartitions = 0;
    public int numberOfTasks = 0;
    public int countPartitionsFinished = 0;
    public double validAvgMs;

    // ========================================================

    private WorkerStatic(int workerId) {
        
        this.workerId = workerId;
        this.model = Dl4jModelFactory.createModel(workerId);
        this.flatModel = Dl4jParamUtils.modelToFlatList(model); 

        this.logger = CustomLogger.getWorkerInstance(workerId);
        if(logger.isEnabled(2)) this.logger.log("Initial Model: " + 
            Dl4jParamUtils.sampleFlat(this.flatModel, SAMPLING_CONSTANT));

        this.pBestWeights = Arrays.copyOf(flatModel, flatModel.length);
        this.stats = new Stats();
        this.psoUpdater = new PsoUpdater(workerId, this);
        this.predictor = new BatchPrediction(model, CustomLogger.getWorkerInstance(workerId)); 
    }
}

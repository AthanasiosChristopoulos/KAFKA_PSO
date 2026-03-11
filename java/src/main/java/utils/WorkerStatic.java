package utils;

import org.apache.kafka.common.serialization.Serdes;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import dl4j_models.Dl4jModelFactory;
import dl4j_models.Dl4jParamUtils;
import dl4j_models.PsoModel;

import org.nd4j.common.primitives.Pair;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.Map;

import message.weights_message.*;
import org.apache.kafka.common.serialization.Serializer;
import org.apache.kafka.common.serialization.Serdes;

import state.*;

public final class WorkerStatic {

    private static final ConcurrentHashMap<Integer, WorkerStatic> INSTANCES = new ConcurrentHashMap<>();

    private static Config cfg = Config.getInstance();
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;

    public final int workerId;

    public final PsoModel model;
    public float[] flatModel;
    public float[] pBestWeights;
    public float[] velocity;

    public final Stats stats;
    public final PsoUpdater psoUpdater;
    public final BatchPrediction predictor;

    private CustomLogger logger;

    public float local_gBestLoss = 100000f;
    public float local_gBestAccuracy = -1f;

    public int batchesRead = 0;
    public int incrementBatchesRead = 0;
    public long incrementSamplesRead = 0;

    public boolean printedReport = false;
    public boolean endedWorker = false;

    public int inactivePartitions = 0;
    public int numberOfTasks = 0;
    public int countPartitionsFinished = 0;
    public double validAvgMs;
    public int countForwardPasses = 0;
    
    public int start;     // configured
    public int headDim;               // number of trainable params in head

    public int pBestCandidateCount = 0; // how many times filter said “send”
    public int pBestForwardedCount = 0;

    public int improved_pBest_count = 0;
    public int significant_pBest_count = 0;

    public long TOTAL_MESSAGES_SENT = 0;
    public long TOTAL_MESSAGES_SENT_PBEST = 0;
    public long TOTAL_MESSAGES_SENT_CURRENT_WEIGHTS = 0;
    public long TOTAL_BYTES_SENT = 0;
    public static long BYTES_PER_WEIGHTSMESSAGE = 0;
    
    public boolean firstActive = false;
    public volatile long simulationStartMs = 0L;

    // Predictor Stuff ==========================================================
    // Predictor comparison experiment: pBest ===================================

    public float[] predictorRefWeightsPBest = null;
    public long predictorTsPBest = -1L;
    public boolean predictorInitializedPBest = false;

    // Predictor comparison experiment: monitoring/current_weights ================
    public long predictorTsMonitoring = -1L;
    public float[] predictorRefWeightsMonitoring = null;
    public boolean predictorInitializedMonitoring = false;

    public float[] predictorRefPsoVelocityMonitoring;

    public TimedWeightsSnapshot snapX1 = null;   // most recent previous monitoring snapshot
    public TimedWeightsSnapshot snapX2 = null;   // one before that
    public TimedWeightsSnapshot snapX3 = null;   // one before that
    public float[] currentVelocityPBest = null;
    public float[] currentVelocityMonitoring = null;

    public MotionTracker motionTracker = new MotionTracker(12);
    public MotionTracker pBestMotionTracker = new MotionTracker(12);

    public long t = 0;

    // ========================================================

    public WorkerStatic(int workerId) {

        INSTANCES.put(workerId, this);
        this.workerId = workerId;
        this.logger = CustomLogger.getWorkerInstance(workerId);
        this.TOTAL_MESSAGES_SENT = 0;
        this.TOTAL_BYTES_SENT = 0;
        // this.model = Dl4jModelFactory.createModel(workerId, false);
        Pair<PsoModel, Integer> pair = Dl4jModelFactory.createModel(workerId, false);

        this.model = pair.getFirst();        // the model
            
        if(cfg.USING_PRETRAINED_MODEL && cfg.FREEZE) {
            
            this.start = pair.getSecond(); // add to Config
            this.headDim = (int) model.numParams() - start;
            if(logger.isEnabled(2)) logger.log("Model Dimensions => " + 
                "start: " + this.start + ", model.numParams(): " + model.numParams() + 
                ", headDim: " + this.headDim);
                
            this.flatModel = Dl4jParamUtils.modelToFlatHead(model, this.start);

        } else {
            this.flatModel = Dl4jParamUtils.modelToFlatList(model); 
        }

        this.velocity = new float[this.flatModel.length];

        if(logger.isEnabled(2)) {
            this.logger.log("Initial Model: " + Dl4jParamUtils.sampleFlat(this.flatModel, SAMPLING_CONSTANT));
            Dl4jParamUtils.saveModel(model, "Init-" + workerId + "-model", this.start);
            logger.log("Model with shape: ");
            Map<String, INDArray> pt = model.paramTable();
            pt.forEach((k, v) -> logger.log(k + " -> " + Arrays.toString(v.shape())));
        }

        this.pBestWeights = Arrays.copyOf(flatModel, flatModel.length);
        initFixedSendSizes(this.flatModel.length);

        this.stats = new Stats();
        this.psoUpdater = new PsoUpdater(workerId, this);
        this.predictor = new BatchPrediction(model, CustomLogger.getWorkerInstance(workerId), this); 

        if(logger.isEnabled(2)) logger.log("ND4J backend: " + Nd4j.getBackend().getClass().getName());
        if(logger.isEnabled(2)) logger.log("Affinity: " + Nd4j.getAffinityManager().getClass().getName());
        if(logger.isEnabled(2)) logger.log("Params buffer class: " + model.params().data().getClass().getName());
        if(logger.isEnabled(2)) logger.log("Params is on device? " + model.params().isAttached());
    }

    // ===========================================================================

    public void incrementTotalMessagesSent(String msgType) {
        TOTAL_MESSAGES_SENT++;
        TOTAL_BYTES_SENT += BYTES_PER_WEIGHTSMESSAGE;
        if(msgType.equals("current_weights")) {
             TOTAL_MESSAGES_SENT_CURRENT_WEIGHTS++;
        } else if(msgType.equals("pBest")) {
            TOTAL_MESSAGES_SENT_PBEST++;
        } else {
            System.out.println("Error Right here 1");
        }

    }

    // ===========================================================================

    public static void initFixedSendSizes(int weightsDim) {

        float[] dummyWeights = new float[weightsDim]; // zeros
        WeightsMessage dummy = new WeightsMessage(0, "00000000-0000-0000-0000-000000000000", 0f, 0f, dummyWeights);
        WeightsMessageSerializer valueSer = new WeightsMessageSerializer();
        byte[] val = valueSer.serialize(null, dummy);
        valueSer.close();
        BYTES_PER_WEIGHTSMESSAGE = val.length;
    }
    
    // =========================================================================================================

    public static String printBatchesReadSummary() {
        long sum = 0;
        long sum_samples = 0;

        StringBuilder sb = new StringBuilder();
        sb.append("[WorkerStatic] batchesRead per instance:\n");

        for (Map.Entry<Integer, WorkerStatic> e : INSTANCES.entrySet()) {
            int wid = e.getKey();
            WorkerStatic ws = e.getValue();
            int br = ws.incrementBatchesRead;
            long sr = ws.incrementSamplesRead;

            sum += br; 
            sum_samples += sr;

            sb.append("  workerId=").append(wid)
            .append(" incrementBatchesRead=").append(br)
            .append(" incrementSamplesRead=").append(sr)
            .append(" tasks=").append(ws.numberOfTasks)
            .append(" inactivePartitions=").append(ws.inactivePartitions)
            .append('\n');
        }

        sb.append("[WorkerStatic] SUM(batchesRead = ").append(sum).append(", samplesRead = ").append(sum_samples)
        .append(") across ").append(INSTANCES.size()).append(" instances");

        return sb.toString();
}
}

// paramTable(): =========================================================================================================
// Counting of layers with numbers. All the Layers get a number, but some mght not have any params (they arent printing)
// Essentially its says for the 3rd layer we have these params: 
// 0: ConvolutionLayer (has params: 0_W, 0_b)
// 1: ActivationLayer (no params)
// 2: SubsamplingLayer (no params)
// 3: ConvolutionLayer (has params: 3_W, 3_b)
// 4: ActivationLayer (no params)
// 5: GlobalPoolingLayer (no params)
// 6: OutputLayer (has params: 6_W, 6_b)
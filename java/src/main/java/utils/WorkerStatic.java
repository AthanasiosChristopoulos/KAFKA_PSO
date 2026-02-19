package utils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.common.primitives.Pair;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Map;

import state.*;

public final class WorkerStatic {

    private static Config cfg = Config.getInstance();
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;

    public final int workerId;

    public final PsoModel model;
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

    public int start;     // configured
    public int headDim;               // number of trainable params in head

    // ========================================================

    public WorkerStatic(int workerId) {
        
        this.workerId = workerId;
        this.logger = CustomLogger.getWorkerInstance(workerId);

        // this.model = Dl4jModelFactory.createModel(workerId, false);
        Pair<PsoModel, Integer> pair = Dl4jModelFactory.createModel(workerId, false);

        this.model = pair.getFirst();        // the model

        try {
            this.start = pair.getSecond(); // add to Config
            this.headDim = (int) model.numParams() - start;
            if(logger.isEnabled(2)) logger.log("Model Dimensions => " + 
                "start: " + this.start + ", model.numParams(): " + model.numParams() + 
                ", headDim: " + this.headDim);
                
            if(cfg.USING_PRETRAINED_MODEL) {
                this.flatModel = Dl4jParamUtils.modelToFlatHead(model, this.start);
            } else {
                this.flatModel = Dl4jParamUtils.modelToFlatList(model); 
            }
        } catch(Exception e) {
            e.printStackTrace();  
        }
        
        if(logger.isEnabled(2)) {
            this.logger.log("Initial Model: " + Dl4jParamUtils.sampleFlat(this.flatModel, SAMPLING_CONSTANT));
            Dl4jParamUtils.saveModel(model, "Init-" + workerId + "-model", this.start);
            logger.log("Model with shape: ");
            Map<String, INDArray> pt = model.paramTable();
            pt.forEach((k, v) -> logger.log(k + " -> " + Arrays.toString(v.shape())));
        }

        this.pBestWeights = Arrays.copyOf(flatModel, flatModel.length);
        this.stats = new Stats();
        this.psoUpdater = new PsoUpdater(workerId, this);
        this.predictor = new BatchPrediction(model, CustomLogger.getWorkerInstance(workerId), this); 
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
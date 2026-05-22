
package state;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling2d;
import dl4j_models.Dl4jParamUtils;
import dl4j_models.PsoModel;

import java.util.ArrayList;
import java.util.List;

import utils.*;
import message.data_message.*; 

public class BatchPrediction {

    private static final Config cfg = Config.getInstance();
    public final int NUM_FEATURES = cfg.NUM_FEATURES;
    public final int NUM_CLASSES = cfg.NUM_CLASSES;
    public final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;
    public final String DATASET = cfg.DATASET;
    public final boolean MEMORY_EFFICIENT = cfg.MEMORY_EFFICIENT;

    private static BatchPrediction coordinatorInstance = null;

    private final CustomLogger logger;

    private final WorkerStatic ws;

    private long start = System.nanoTime();
    private long end = System.nanoTime();

    private final boolean MODEL_IS_CNN;
    private INDArray X4d;
    private INDArray X2d;
    private INDArray X;
    private INDArray logits_probs; 
    private INDArray argMax;

    private float EPS = 0.0000001f; 

    List<float[]> featureList = new ArrayList<>();
    List<Integer> labels = new ArrayList<>();
    float[] probabilities;

    private final boolean coordinator;
    private PsoModel coordinatorModel;
    public float[] slopeLambdas;
    public int workerId = -1;

    // for Worker =====================================================================

    public BatchPrediction(PsoModel model, CustomLogger logger, WorkerStatic ws) {

        this.ws = ws;
        this.workerId = ws.workerId;
        this.MODEL_IS_CNN = model.isCnn();
        this.logger = logger;
        this.coordinator = false;
    }

    // for Coordinator ==================================================================================================

    public BatchPrediction(PsoModel model, PsoModel bestModel, CustomLogger logger) {

        this.ws = null;
        this.coordinatorModel = model;
        this.MODEL_IS_CNN = model.isCnn();
        this.logger = logger;
        this.coordinator = true;
    }

    // ===========================================================================

    public static BatchPrediction getInstanceForCoordinator(PsoModel model, PsoModel bestModel, CustomLogger logger) {

        if(coordinatorInstance == null) {
            coordinatorInstance = new BatchPrediction(model, bestModel, logger);
            return coordinatorInstance;
        } 

        return coordinatorInstance;
    }

    // ===========================================================================

    public float[] callPredictionsBatch(List<DataMessage> batch, PsoModel argument_model, boolean deleteWorkspace) {

        if(MEMORY_EFFICIENT) GpuMem.log("[Worker " + workerId + "] START");

        if (batch == null || batch.isEmpty()) {
            if (logger.isEnabled(2)) logger.log("Batch is empty");
            return null;
        }

        for (DataMessage msg : batch) {
            if (msg == null) continue;

            float[] features = msg.features;    // (NUM_FEATURES,)   
     
            if (features == null) {
                if (logger.isEnabled(2)) logger.log("Null features in DataMessage");
                continue;
            }

            if (features.length != NUM_FEATURES) {
                if (logger.isEnabled(2)) logger.log("Wrong features length. Got " + features.length + " but NUM_FEATURES = " + NUM_FEATURES);
                System.out.println("Wrong features length. Got " + features.length + " but NUM_FEATURES = " + NUM_FEATURES);
                continue; 
            }

            featureList.add(features); 
            labels.add(msg.label);
        }

        int nSamples = featureList.size();
        if (nSamples == 0) {
            if (logger.isEnabled(2)) logger.log("No samples after parsing batch");
            return null;
        }

        float[][] data = new float[nSamples][NUM_FEATURES];     // matrix of samples and features
                                                                // (nSamples, NUM_FEATURES)
        for (int i = 0; i < nSamples; i++) {
            System.arraycopy(featureList.get(i), 0, data[i], 0, NUM_FEATURES);
        }

        // Forward Pass Start ===============================================================================
        // in this part, we need to unflatten the data input in case that it is CNN

        if(MODEL_IS_CNN) {

            if(DATASET.contains("cifar") || DATASET.contains("svhn")) {

                X2d = Nd4j.create(data);                    // [batch, 3072] => 3 * 32 * 32 = 3072
                                                            // (nSamples, 3072)
                X4d = X2d.reshape(nSamples, 32, 32, 3);     // [batch, 32, 32, 3]
                                                            // (nSamples, 32, 32, 3)
                if (argument_model.isNhWC()) {
                    X = X4d;                                // keep NHWC

                    if(cfg.TRANSFORM_IMAGE) {
                        INDArray X224_nhwc = Nd4j.exec(new Upsampling2d(X4d, 7, 7, false))[0];
                        X224_nhwc = X224_nhwc.div(127.5).sub(1.0);
                        X = X224_nhwc;
                    }
                    
                } else {
                    X = X4d.permute(0, 3, 1, 2);         // convert to NCHW
                }                                        // (nSamples, 3, 32, 32)


            } else { // else if("mnist".equals(DATASET) || "mnist5".equals(DATASET) ) {

                X2d = Nd4j.create(data);                    // (nSamples, 784)
                X4d = X2d.reshape(nSamples, 28, 28, 1);     // (nSamples, 28, 28, 1)

                if (argument_model.isNhWC()) {
                    X = X4d;                                // keep NHWC (TensorFlow/Keras style)
                } else {
                    X = X4d.permute(0, 3, 1, 2);            // convert to NCHW (DL4J style)
                }   
            }
        } else {    // Normal dataset (no image) + no CNN used 

            X = Nd4j.create(data);                     // (nSamples, NUM_FEATURES)
        }

        // Debugging prints ==========================================================
        // System.out.println("NUM_FEATURES = " + NUM_FEATURES);
        // System.out.println("MODEL_IS_CNN = " + MODEL_IS_CNN);
        // System.out.println("X shape = " + java.util.Arrays.toString(X.shape()));
        // System.out.println("X order = " + X.ordering());
        // System.out.println("Model expects NHWC? " + argument_model.isNhWC());
        // System.out.println("Input shape to model: " + Arrays.toString(X.shape()));
        // System.exit(0);

        if(MEMORY_EFFICIENT) GpuMem.log("[Worker " + workerId + "] BEFORE FORWARD");

        // ====================================================================================

        start = System.nanoTime();  // We only want to evaluate the performance of the forward pass, but this also includes the GPU transfer overhead
        
        try {
            logits_probs = argument_model.output(X, false);    // forward pass (GPU critical point), (nSamples, NUM_CLASSES) or (nSamples, 1) if sigmoid. Here is where the memory transfer happens between CPU and GPU
            Nd4j.getExecutioner().commit();
            
        } catch (Exception e) {
            System.err.println("GPU inference failed on worker " + workerId + ": " + e.getMessage());
            e.printStackTrace();
            CoordinatorControl.getInstance().requestStop(workerId);
            return null;
        }        
        
        // ===================================================================================================
        // Activate this only if Models outs only logits (.activation(Activation.IDENTITY) and HINGE function)
        // if (cfg.NEED_PROBS) {
        //     logits_probs = Nd4j.nn().softmax(logits_probs.dup(), 1);
        // }
        // ===================================================================================================

        if (MEMORY_EFFICIENT && (GpuMem.usedPercent() >= 0.80) || deleteWorkspace) {
            System.out.println("Reducing Memory: Destroying workspaces");
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        }

        end = System.nanoTime();
        if(MEMORY_EFFICIENT) GpuMem.log("[Worker " + workerId + "] AFTER FORWARD");

        // Forward Pass End ===============================================================================

        if (logits_probs == null || logits_probs.size(0) == 0) {
            if (logger.isEnabled(2)) logger.log("Empty logits_probs batch");
            return null;
        }

        int nCorrect = 0;
        float loss = 0f;

        float[] sampleLosses = new float[nSamples];
        int outDim = (int) logits_probs.size(1);

        // Make sure the scratch buffer fits the actual output dimension
        if (probabilities == null || probabilities.length < outDim) {
            probabilities = new float[outDim];
        }

        // BINARY CASE (SIGMOID) ======================================================================

        if (NEURAL_OUTPUT == 1) {

            for (int i = 0; i < nSamples; i++) {

                float p = logits_probs.getFloat(i, 0);

                if (p < EPS) p = EPS;
                if (p > 1f - EPS) p = 1f - EPS;

                int label = labels.get(i);   // 0 or 1

                int pred = (p >= 0.5f) ? 1 : 0;
                if (pred == label) nCorrect++;

                sampleLosses[i] = LossFunction.compute_loss(p, label);
            }

        // MULTI-CLASS CASE (SOFTMAX) =================================================================

        } else {
        
            argMax = logits_probs.argMax(1); 
            Nd4j.getExecutioner().commit();

            float[] flatProps = logits_probs.data().asFloat(); 

            for (int i = 0; i < nSamples; i++) {

                int pred = argMax.getInt(i);    // accuracy is dependent on this and only this not from logits_probs

                int label = labels.get(i);
                if (pred == label) nCorrect++;

                int base = i * outDim;         
                for (int c = 0; c < outDim; c++) {
                    probabilities[c] = flatProps[base + c];     // this will be used to calculate loss
                }
                sampleLosses[i] = LossFunction.compute_loss(probabilities, label);
            }
        }

        // Combine Losses from Multiple Samples =======================================================

        if ("TOP_K".equals(cfg.COMBINE_LOSS)) {
            loss = LossFunction.topKAverage(sampleLosses);

        } else if ("SUM".equals(cfg.COMBINE_LOSS)) {
            loss = LossFunction.sum(sampleLosses);

        } else if ("AVG".equals(cfg.COMBINE_LOSS)) {
            loss = LossFunction.average(sampleLosses);

        } else {
            loss = LossFunction.average(sampleLosses);
        }

        // Regularization Cost =======================================================

        if (!"NONE".equals(cfg.REGULARIZER)) {

            if (!this.coordinator) {    // the regularization cost only is taking into account on the worker

                if ("L2".equals(cfg.REGULARIZER)) {
                    loss += (float) (cfg.LAMBDA_VALUE * LossFunction.l2Penalty(ws.flatModel));

                } else if ("GROUP_LASSO".equals(cfg.REGULARIZER) && argument_model.asMultiLayerNetwork() != null) {
                    loss += (float) (cfg.LAMBDA_VALUE * LossFunction.groupLassoNeuronPenalty(argument_model.asMultiLayerNetwork(), true));

                }  else if ("SLOPE".equals(cfg.REGULARIZER)) {

                    if (slopeLambdas == null || slopeLambdas.length != ws.flatModel.length) {
                        slopeLambdas = LossFunction.makeSlopeLambdasGeometric(ws.flatModel.length,
                            1e-2f, 0.995f);  // 0.995f means a slow decay, but it must be smaller than 1
                            // makeSlopeLambdasGeometric(int d, float lambda1, float alpha)
                    }

                    loss += (float) (cfg.LAMBDA_VALUE * LossFunction.slopePenalty(ws.flatModel, slopeLambdas));
                
                } else if ("NONE".equals(cfg.REGULARIZER)) {
                    // no weight penalties applied
                }


            } else {

                float[] flatCoordinatorModel = Dl4jParamUtils.modelToFlatList(this.coordinatorModel);

                if ("L2".equals(cfg.REGULARIZER)) {
                    loss += (float) (cfg.LAMBDA_VALUE * LossFunction.l2Penalty(flatCoordinatorModel));

                } else if ("GROUP_LASSO".equals(cfg.REGULARIZER) && argument_model.asMultiLayerNetwork() != null) {
                    loss += (float) (cfg.LAMBDA_VALUE * LossFunction.groupLassoNeuronPenalty(argument_model.asMultiLayerNetwork(), true));

                }  else if ("SLOPE".equals(cfg.REGULARIZER)) {

                    if (slopeLambdas == null || slopeLambdas.length != flatCoordinatorModel.length) {
                        slopeLambdas = LossFunction.makeSlopeLambdasGeometric(flatCoordinatorModel.length,
                            1e-2f, 0.995f);  // 0.995f means a slow decay, but it must be smaller than 1
                            // makeSlopeLambdasGeometric(int d, float lambda1, float alpha)
                    }

                    loss += (float) (cfg.LAMBDA_VALUE * LossFunction.slopePenalty(flatCoordinatorModel, slopeLambdas));
                
                } else if ("NONE".equals(cfg.REGULARIZER)) {
                    // no weight penalties applied
                }
            }
        }

        // ===========================================================================================
        
        if (Float.isNaN(loss) || Float.isInfinite(loss)) {
            if (logger.isEnabled(2)) logger.log("loss is NaN/Inf, X length: " + X.length());
            return null;
        }

        if (nCorrect == 0) {
            if (logger.isEnabled(2)) logger.log("nCorrect == 0");
        }

        float accuracy = (float) nCorrect / nSamples;

        float forwardMs = (end - start) / 1_000_000f;

        featureList.clear();
        labels.clear();

        if(MEMORY_EFFICIENT) GpuMem.log("[Worker " + workerId + "] AFTER CLOSE");

        return new float[]{accuracy, loss, nSamples, nCorrect, forwardMs};

    }
}




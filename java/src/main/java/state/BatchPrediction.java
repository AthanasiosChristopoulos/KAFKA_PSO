
package state;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling2d;

import dl4j_models.PsoModel;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.Arrays;

import utils.*;
import message.data_message.*; 
import message.weights_message.*; 

import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;

public class BatchPrediction {

    private static final Config cfg = Config.getInstance();
    public final int NUM_FEATURES = cfg.NUM_FEATURES;
    public final int NUM_CLASSES = cfg.NUM_CLASSES;
    public final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;

    public final String DATASET = cfg.DATASET;
    private static final String LOSS_FUNCTION = cfg.LOSS_FUNCTION;
    private static final String COMBINE_LOSS = cfg.COMBINE_LOSS;
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
    private INDArray Xbuffer;
    private INDArray logits_probs; 
    private INDArray argMax;

    private int EXPECTED_SIZE; 
    private float EPS = 0.0000001f; 

    List<float[]> featureList = new ArrayList<>();
    List<Integer> labels = new ArrayList<>();
    // float[] probabilities = new float[NUM_CLASSES];
    float[] probabilities;

    private final boolean coordinator;

    private transient MemoryWorkspace inferenceWsObj;
    private transient String wsName;

    private static final WorkspaceConfiguration WS_CONF =
        WorkspaceConfiguration.builder()
            .initialSize(0)
            .overallocationLimit(0) // <--- key: no overalloc bursts
            .policyAllocation(AllocationPolicy.STRICT) // <--- key: don't overallocate
            .policyLearning(LearningPolicy.NONE) // <--- key: don't keep learning new bigger sizes
            .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
            .policySpill(SpillPolicy.REALLOCATE)
            .policyMirroring(MirroringPolicy.FULL)
            .build();

    public float[] slopeLambdas;
    public int workerId = -1;

    // for Worker (from WorkerStatic) =======================================================================================================

    public BatchPrediction(PsoModel model, CustomLogger logger, WorkerStatic ws) {
        this.ws = ws;
        this.workerId = ws.workerId;
        this.MODEL_IS_CNN = model.isCnn();
        this.logger = logger;
        EXPECTED_SIZE = cfg.BATCH_SIZE;

        if (MODEL_IS_CNN) {     // Instance Xbuffer based on nature / dimensionality of input data
            if (DATASET.contains("mnist")) {
                Xbuffer = Nd4j.create(EXPECTED_SIZE, 1, 28, 28);
            } else if (DATASET.contains("cifar") || DATASET.contains("svhn")) {
                if (model.isNhWC()) Xbuffer = Nd4j.create(EXPECTED_SIZE, 32, 32, 3);
                else               Xbuffer = Nd4j.create(EXPECTED_SIZE, 3, 32, 32);
            }
        } else {
            Xbuffer = Nd4j.create(EXPECTED_SIZE, NUM_FEATURES);
        }
        this.coordinator = false;
        wsName = "INFERENCE_WS_" + workerId;
    }

    // for Coordinator ==================================================================================================

    public BatchPrediction(PsoModel model, PsoModel bestModel, CustomLogger logger) {

        this.ws = null;

        this.MODEL_IS_CNN = model.isCnn();
        this.logger = logger;
        this.EXPECTED_SIZE = 500;

        if (MODEL_IS_CNN) {
            if (DATASET.contains("mnist")) {
                Xbuffer = Nd4j.create(EXPECTED_SIZE, 1, 28, 28);
            } else if (DATASET.contains("cifar") || DATASET.contains("svhn")) {
                if (model.isNhWC()) Xbuffer = Nd4j.create(EXPECTED_SIZE, 32, 32, 3);
                else Xbuffer = Nd4j.create(EXPECTED_SIZE, 3, 32, 32);
            }
        } else {
            Xbuffer = Nd4j.create(EXPECTED_SIZE, NUM_FEATURES);
        }

        this.coordinator = true;
        wsName = "INFERENCE_WS_" + workerId;
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

    // public INDArray outputWithWorkspace(PsoModel model, INDArray x) {
    //     try (MemoryWorkspace ws = Nd4j.getWorkspaceManager()
    //             .getAndActivateWorkspace(WS_CONF, "INFERENCE_WS")) {

    //         INDArray y = model.output(x, false);

    //         INDArray yDetached = y.detach(); // safe copy out of workspace
    //         Nd4j.getExecutioner().commit();
    //         return yDetached;
    //     }
    // }

    // public INDArray outputWithWorkspace(PsoModel model, INDArray x) {
    //     INDArray y;
    //     try (MemoryWorkspace ws = Nd4j.getWorkspaceManager()
    //             .getAndActivateWorkspace(WS_CONF, "INFERENCE_WS")) {

    //         y = model.output(x, false);

    //         Nd4j.getExecutioner().commit();
            
    //     }
    //     return y;
    // }

    // public INDArray outputWithWorkspace(PsoModel model, INDArray x) {
    //     try (MemoryWorkspace ws = Nd4j.getWorkspaceManager()
    //             .getAndActivateWorkspace(WS_CONF, "INFERENCE_WS_" + this.workerId)) {

    //         INDArray y = model.asMultiLayerNetwork().output(x, false, ws);
    //         Nd4j.getExecutioner().commit();
    //         logits_probs = y;
    //         return y.detach(); // valid ONLY while ws is still open
    //     }
    // }

    // ===========================================================================

    public INDArray outputWithWorkspace(PsoModel model, INDArray x) {
        MemoryWorkspace ws = null;
        try (MemoryWorkspace w = Nd4j.getWorkspaceManager()
                .getAndActivateWorkspace(WS_CONF, wsName)) {

            ws = w; // keep reference to the same object
            INDArray y = model.asMultiLayerNetwork().output(x, false, w);
            Nd4j.getExecutioner().commit();
            return y.detach();

        } finally {
            // ws is now CLOSED (try-with-resources already ran close())
            if (ws != null && GpuMem.freeMb() >= 0 && GpuMem.freeMb() < 2000) {
                System.out.println("Reducing Memory: destroying workspace " + wsName);
                Nd4j.getWorkspaceManager().destroyWorkspace(ws);
            }
        }
    }
    
    // ===========================================================================

    private MemoryWorkspace getInferenceWsObj() {
        if (wsName == null) wsName = "INFERENCE_WS_" + workerId;
        // activate to create it, then close right away so it's not active
        MemoryWorkspace ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(WS_CONF, wsName);
        ws.close();
        inferenceWsObj = ws;
        return inferenceWsObj;
    }
    // ===========================================================================

    public void maybeDestroyInferenceWorkspaceIfLowMem() {
        if (GpuMem.freeMb() < 1200) {
            System.out.println("Reducing Memory");
            MemoryWorkspace ws = getInferenceWsObj();
            Nd4j.getWorkspaceManager().destroyWorkspace(ws);
            inferenceWsObj = null; // will recreate later
        }
    }
    // ===========================================================================

    // public INDArray forwardOnce(PsoModel model, INDArray x) {
    //     try (MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
    //         // This guarantees "no workspace active", so outputOfLayersDetached won't throw
    //         return model.output(x, false);
    //     }
    // }
    // ===========================================================================

    public float[] callPredictionsBatch(List<DataMessage> batch, PsoModel argument_model, boolean deleteWorkspace) {

        if(MEMORY_EFFICIENT) GpuMem.log("[Worker " + workerId + "] START");

        if (batch == null || batch.isEmpty()) {
            if (logger.isEnabled(2)) logger.log("Batch is empty");
            return null;
        }
        // logger.log("batch size of: " + batch.size());

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
                continue; // skip non-conforming record
            }

            featureList.add(features);  // of nSamples. (nSamples, NUM_FEATURES)
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

        if(true || !MEMORY_EFFICIENT) {
            
            // ==============================================================================================================
            // Alternative 1) Costs Memory (Allocates new Memory every time), but Better Time and simplicity
            // doesnt seem to have a significant difference memory wise
            // choose this for better performance and simpler code
            // X2d = Nd4j.create(data); X = X2d.reshape(...); X = X4d.permute(...); happens on the CPU memory / host-side NDArray
            // costs around 22 ms Predicition cost (mostly the forward pass, meaning the competion of the workers over the GPU) 
            
            // in this part, we need to unflatten the data input in case that it is CNN
            if(MODEL_IS_CNN) {

                if(DATASET.contains("cifar") || DATASET.contains("svhn")) {

                    X2d = Nd4j.create(data);                       // [batch, 3072] => 3 * 32 * 32 = 3072
                        // (nSamples, 3072)
                    X4d = X2d.reshape(nSamples, 32, 32, 3);        // [batch, 32, 32, 3]
                        // (nSamples, 32, 32, 3)
                    if (argument_model.isNhWC()) {
                        // System.out.println("AAAAAAAAAAAAAAAAAA");
                        X = X4d;                             // keep NHWC

                        if(cfg.TRANSFORM_IMAGE) {
                            System.out.println("I am TRANSFORM_IMAGE");
                            INDArray X224_nhwc = Nd4j.exec(new Upsampling2d(X4d, 7, 7, false))[0];
                            X224_nhwc = X224_nhwc.div(127.5).sub(1.0);
                            X = X224_nhwc;
                        }
                        
                    } else {
                        // System.out.println("BBBBBBBBBBBBBBBBBB");
                        X = X4d.permute(0, 3, 1, 2);         // convert to NCHW
                    }        // (nSamples, 3, 32, 32)


                } else { // else if("mnist".equals(DATASET) || "mnist5".equals(DATASET) ) {

                    X2d = Nd4j.create(data);          // (nSamples, 784)
                    
                    X4d = X2d.reshape(nSamples, 28, 28, 1);  // (nSamples, 28, 28, 1)

                    if (argument_model.isNhWC()) {
                        // System.out.println("AAAAAAAAAAAAAAAAAA");
                        X = X4d;                             // keep NHWC (TensorFlow/Keras style)
                    } else {
                        // System.out.println("BBBBBBBBBBBBBBBBBB");
                        X = X4d.permute(0, 3, 1, 2);          // convert to NCHW (DL4J style)
                    }   
                }

            } else {    // Normal dataset (no image) + no CNN used 

                X = Nd4j.create(data);                     // (nSamples, NUM_FEATURES)
            }

        } else {

            // ==============================================================================================================
            // Alternative 2)

            if (nSamples > EXPECTED_SIZE) {
                if (logger.isEnabled(2)) logger.log("Batch bigger than EXPECTED_SIZE: nSamples=" + nSamples + " EXPECTED_SIZE=" + EXPECTED_SIZE + " -> clipping");
                System.out.println("Batch bigger than EXPECTED_SIZE: nSamples=" + nSamples + " EXPECTED_SIZE=" + EXPECTED_SIZE + " -> clipping");

                nSamples = EXPECTED_SIZE;
            }

            for (int i = 0; i < nSamples; i++) {

                float[] features = featureList.get(i);

                if (MODEL_IS_CNN) {

                    if (DATASET.contains("mnist")) {

                        // flatten 784 into 1x28x28 => is basically 2D
                        // this is row-major mapping (C-order), meaning the columns change fastest
                        for (int j = 0; j < NUM_FEATURES; j++) {
                            int row = j / 28;   // integer division
                            int col = j % 28;   // change fastest
                            Xbuffer.putScalar(new int[]{i, 0, row, col}, features[j]);
                        }

                    } else if (DATASET.contains("cifar") || DATASET.contains("svhn")) {

                        for (int j = 0; j < NUM_FEATURES; j++) {
                            // int channel = j / (32 * 32);
                            // int pixel = j % (32 * 32);
                            // int row = pixel / 32;
                            // int col = pixel % 32;
                            int pixel   = j / 3;     // 0..1023
                            int channel = j % 3;     // 0..2, fastest changing, j changes ecery loop
                            int row     = pixel / 32;
                            int col     = pixel % 32;  // column changes every 3 loop (look at pixel)

                            if (argument_model.isNhWC()) {
                                Xbuffer.putScalar(new int[]{i, row, col, channel}, features[j]);
                                // do ordering, as the data 
                            } else {
                                Xbuffer.putScalar(new int[]{i, channel, row, col}, features[j]);
                                // do ordering differently from input data
                            }
                                
                        }
                    }

                } else {
                    for (int j = 0; j < NUM_FEATURES; j++) {
                        Xbuffer.putScalar(i, j, features[j]);
                    }
                }
            }
            // float[] xb = Xbuffer.data().asFloat();
            // float before = Xbuffer.getFloat(0, 0, 0, 0);
            // xb[0] = 123.456f;
            // float after  = Xbuffer.getFloat(0, 0, 0, 0);

            // System.out.println("before=" + before + " after=" + after);
            // for (int i = 0; i < nSamples; i++) {
            //     float[] features = featureList.get(i);  // length 784
            //     System.arraycopy(features, 0, xb, i * NUM_FEATURES, NUM_FEATURES);
            // }

            // if (nSamples == EXPECTED_SIZE) {
            //     X = Xbuffer;
            // } else {
            //     X = Xbuffer.get(
            //         NDArrayIndex.interval(0, nSamples),
            //         NDArrayIndex.all(),
            //         NDArrayIndex.all(),
            //         NDArrayIndex.all()
            //     );
            // }

            if (nSamples == EXPECTED_SIZE) {
                X = Xbuffer;
            } else {
                if (MODEL_IS_CNN) {
                    if ((DATASET.contains("cifar") || DATASET.contains("svhn")) && argument_model.isNhWC()) {
                        X = Xbuffer.get(NDArrayIndex.interval(0, nSamples),
                                        NDArrayIndex.all(),
                                        NDArrayIndex.all(),
                                        NDArrayIndex.all());
                    } else if (DATASET.contains("cifar") || DATASET.contains("svhn")) {
                        X = Xbuffer.get(NDArrayIndex.interval(0, nSamples),
                                        NDArrayIndex.all(),
                                        NDArrayIndex.all(),
                                        NDArrayIndex.all());
                    } else { // MNIST NCHW
                        X = Xbuffer.get(NDArrayIndex.interval(0, nSamples),
                                        NDArrayIndex.all(),
                                        NDArrayIndex.all(),
                                        NDArrayIndex.all());
                    }
                } else {
                    X = Xbuffer.get(NDArrayIndex.interval(0, nSamples), NDArrayIndex.all());
                }
            }
        }
   
        // System.out.println("NUM_FEATURES = " + NUM_FEATURES);
        // System.out.println("MODEL_IS_CNN = " + MODEL_IS_CNN);
        // System.out.println("X shape = " + java.util.Arrays.toString(X.shape()));
        // System.out.println("X order = " + X.ordering());
        // System.out.println("Model expects NHWC? " + argument_model.isNhWC());

        if(MEMORY_EFFICIENT) GpuMem.log("[Worker " + workerId + "] BEFORE FORWARD");

        // Evaluate input shape ========================================================
        // System.out.println("Input shape to model: " + Arrays.toString(X.shape()));
        // System.exit(0);
        // ==============================================================================================================

        start = System.nanoTime();                // We only want to evaluate the performance of the forward pass, but this also includes the GPU transfer overhead
        logits_probs = argument_model.output(X, false);    // (nSamples, NUM_CLASSES) or (nSamples, 1) if sigmoid. Here is where the memory transfer happens between CPU and GPU
        // logits_probs = GpuGate.outputExclusive(argument_model, X, workerId);
        // logits_probs = outputWithWorkspace(argument_model, X); 
    
        // if (cfg.NEED_PROBS) {
        //     logits_probs = Nd4j.nn().softmax(logits_probs.dup(), 1);
        // }

        Nd4j.getExecutioner().commit();

        // this is one forward pass per batch (has multiple samples), X is one of the different 
        // dimensionalities identified above. This allocates memory by it self
        // model.output(X,false) is the core inference forward pass.

        // logits_probs = forwardOnce(argument_model, X);
        // Nd4j.getExecutioner().commit(); 
        
        // Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        // Nd4j.getMemoryManager().purgeCaches();

        if (MEMORY_EFFICIENT && (GpuMem.usedPercent() >= 0.80) || deleteWorkspace) {
            System.out.println("Reducing Memory: Destroying workspaces");
            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        }

        end = System.nanoTime();

        // maybeDestroyInferenceWorkspaceIfLowMem();
        // double used = GpuMem.usedMb();
        // if (used > 2800) { // pick a threshold
        //     System.out.println("Release the workspace");
        //     // Nd4j.getWorkspaceManager().destroyWorkspace("INFERENCE_WS");
        //     MemoryWorkspace toDestroy = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("INFERENCE_WS_" + workerId);
        //     Nd4j.getWorkspaceManager().destroyWorkspace(toDestroy);
        // }
        // double min = logits_probs.minNumber().doubleValue();
        // double max = logits_probs.maxNumber().doubleValue();
        // System.out.println("logits_probs min/max = " + min + " / " + max);
        
        // GpuMem.log("[Worker " + workerId + " - " +  Thread.currentThread().getName() + "] AFTER FORWARD");
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
        // System.out.println("outDim: " + outDim + ", logits_probs: " + logits_probs);
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

            if(!LOSS_FUNCTION.equals("CROSS_ENTROPY") || true) {
                // System.out.println("NUM_CLASSES: " + outDim);
            
                argMax = logits_probs.argMax(1);   // max probability => this is what we are deciding
                Nd4j.getExecutioner().commit();

                float[] flatProps = logits_probs.data().asFloat();  // row-major view of logits_probs data

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
                // for (int i = 0; i < nSamples; i++) {

                //     int pred = argMax.getInt(i);
                //     int label = labels.get(i);
                //     if (pred == label) nCorrect++;

                //     // float[] probabilities = logits_probs.getRow(i).toFloatVector();   
                //         // Expensive because: getRow(i) creates a view,
                //         // toFloatVector() allocates a new float[] and copies data every iteration
                //     // float p = probabilities[label];
                //     // if (p < EPS) p = EPS;

                //     // probabilities = logits_probs.data().asFloat();
                //     INDArray row = logits_probs.getRow(i);     // still creates a view per sample
                //     for (int c = 0; c < NUM_CLASSES; c++) {
                //         probabilities[c] = row.getFloat(c);     // doesnt allocate but overwrites memory
                //     }
                //     sampleLosses[i] = LossFunction.compute_loss(probabilities, label);
                    
                //     // float p1 = logits_probs.getFloat(i, label);
                //     // logger.log("Label = " + label + ", probabilities: " + 
                //     //     Arrays.toString(probabilities) + ", p = " + p1);

                //     // if(p1 < EPS) {
                //     //     p1 = EPS;
                //     // }

                //     // sampleLosses[i] = (float) -Math.log(p1);
                // }

                // // logger.log("sampleLosses = " + Arrays.toString(sampleLosses) + 
                // //             "sampleLosses_fast = " + Arrays.toString(sampleLosses_fast));
                        
                // // logger.log("Same arrays? " + Arrays.equals(sampleLosses, sampleLosses_fast));


            } else {
                
                argMax = logits_probs.argMax(1);

                for (int i = 0; i < nSamples; i++) {
                    int pred = argMax.getInt(i);
                    int label = labels.get(i);
                    if (pred == label) nCorrect++;

                    float p = logits_probs.getFloat(i, label);     // logits_probs.getFloat(i, j), logits_probs is 2D-Array 
                                                            // logits_probs = [nSamples, NUM_CLASSES]
                    if(p < EPS) p = EPS;
                    
                    sampleLosses[i] = (float) -Math.log(p); // cross-entropy
                }
            }
        }
        // end = System.nanoTime();    // this is where the forward pass reliably ends due to gathering the logits_probs 
        // too and having forced a stnc between the GPU and CPU
        // Combine Losses from Multiple Samples =======================================================

        if ("TOP_K".equals(COMBINE_LOSS)) {
            loss = LossFunction.topKAverage(sampleLosses);

        } else if ("SUM".equals(COMBINE_LOSS)) {
            loss = LossFunction.sum(sampleLosses);

        } else if ("AVG".equals(COMBINE_LOSS)) {
            loss = LossFunction.average(sampleLosses);

        } else {
            loss = LossFunction.average(sampleLosses);
        }

        // Regularization Cost =======================================================
    
        if (!this.coordinator) {

            if ("L2".equals(cfg.REGULARIZER)) {
                loss += (float) (cfg.LAMBDA_VALUE * LossFunction.l2Penalty(ws.flatModel));

            } else if ("GROUP_LASSO".equals(cfg.REGULARIZER) && argument_model.asMultiLayerNetwork() != null) {
                loss += (float) (cfg.LAMBDA_VALUE * LossFunction.groupLassoNeuronPenalty(argument_model.asMultiLayerNetwork(), true));

            }  else if ("SLOPE".equals(cfg.REGULARIZER)) {

                if (slopeLambdas == null || slopeLambdas.length != ws.flatModel.length) {
                    slopeLambdas = LossFunction.makeSlopeLambdasGeometric(ws.flatModel.length,
                        1e-3f,0.995f);  // 0.995f means a slow decay, but it must be smaller than 1
                }

                loss += (float) (cfg.LAMBDA_VALUE * LossFunction.slopePenalty(ws.flatModel, slopeLambdas));
            
            } else if ("NONE".equals(cfg.REGULARIZER)) {
                // ...
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
        // X.close();
        // logits_probs.close();
        featureList.clear();
        labels.clear();

        if(false && !MEMORY_EFFICIENT) {
            if (logits_probs != null) logits_probs.close();
            if (X != null && X != Xbuffer) X.close();
            if (argMax != null) argMax.close();
            if (X2d != null) X2d.close();
            if (X4d != null) X4d.close();
        }

        // GpuMem.log("[Worker " + workerId + " - " +  Thread.currentThread().getName() + "] AFTER CLOSE");
        if(MEMORY_EFFICIENT) GpuMem.log("[Worker " + workerId + "] AFTER CLOSE");

        return new float[]{accuracy, loss, nSamples, nCorrect, forwardMs};

    }

        
    // ===========================================================================

    // public String predictSingleBest(DataMessage msg) {
    //     if (msg == null || msg.features == null || msg.features.length != NUM_FEATURES) {
    //         return null;
    //     }

    //     try {
    //         INDArray X = Nd4j.create(msg.features).reshape(1, NUM_FEATURES);
    //         INDArray logits_probs = bestModel.output(X, false);
    //         int pred = logits_probs.argMax(1).getInt(0);

    //         Map<String, Object> out = new HashMap<>();
    //         out.put("sample_index", msg.sampleIndex);
    //         out.put("prediction", pred);

    //         return MAPPER.writeValueAsString(out);

    //     } catch (Exception e) {
    //         if (logger.isEnabled(2)) logger.log("Error in predictSingle: " + e.getMessage());
    //         e.printStackTrace();
    //         return null;
    //     }
    // }
}




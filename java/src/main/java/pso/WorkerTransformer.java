package pso;

import org.apache.kafka.streams.kstream.Transformer;
import org.apache.kafka.streams.KeyValue;
import org.apache.kafka.streams.processor.ProcessorContext;
import org.apache.kafka.streams.processor.PunctuationType;
import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.streams.state.ReadOnlyKeyValueStore;
import org.apache.kafka.streams.state.ValueAndTimestamp;
import org.apache.kafka.streams.state.KeyValueIterator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;

import java.util.*;

import utils.*;
import state.*;
import message.data_message.*; 
import message.weights_message.*;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.kafka.streams.processor.PunctuationType;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

public class WorkerTransformer implements Transformer<String, DataMessage, KeyValue<String, WeightsMessage>> {

    private final int workerId;

    private static Config cfg = Config.getInstance();
    private final int TRAIN_SIZE = cfg.TRAIN_SIZE;
    private final int N_BATCHES = cfg.N_BATCHES;  
    private final boolean FULLY_INFORMED = cfg.FULLY_INFORMED;
    private final boolean FILTER_ENABLED = cfg.FILTER_ENABLED;
    private final float SIGNIFICANT_LOSS_DIFF = cfg.SIGNIFICANT_LOSS_DIFF;
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;
    private final float CONVERGENCE_ALPHA = cfg.CONVERGENCE_ALPHA;

    private ProcessorContext context;

    private ReadOnlyKeyValueStore<String, ValueAndTimestamp<WeightsMessage>> bestStore;

    private final List<DataMessage> buffer = new ArrayList<>();

    // private static final MultiLayerNetwork model = Dl4jModelFactory.createModel();;
    // private static final Stats stats = new Stats();;
    // private static BatchPrediction predictor;
    // private static PsoUpdater psoUpdater;

    // private static float local_gBestAccuracy = -1f;
    // private static float local_gBestLoss = 10000f;

    private CustomLogger logger;
    private final WorkerStatic ws;
    
    private float[] pBestWeights;
    private int batchesRead = 0;
    
    private float[] velocity;

    private boolean printedOffset = false;

    private String stateStoreName;
    private String keyName;


    private float accuracy = -1f;
    private float loss = 10000f;
    private int nSamples = 0;
    private int nCorrect = 0;

    private long t0;
    private final AtomicLong t1;
    private double lastActivitySeconds = 0.0;
    
    private long start = System.nanoTime();
    private long end = System.nanoTime();
    private long sumElapsedNs = 0;
    private int count = 0;
    private float forwardPassNs = 0;
    private int countForwardPass = 0;

    private final Set<Integer> seenPartitions = ConcurrentHashMap.newKeySet();
    private long lastOffset = 0;
    private double eps = 1e-12;

    private static final long IDLE_MS = 3000; 
    private static final long CHECK_EVERY_MS = 100; // how often we check
    private static final long IDLE_GRACE_MS = 5000;

    private static final AtomicInteger INSTANCE_SEQ = new AtomicInteger(0);
    private final int instanceNo = INSTANCE_SEQ.incrementAndGet();
    private final String taskInstance = instanceNo + "@" + Integer.toHexString(System.identityHashCode(this));

    private final CoordinatorControl control; 

    private int bufferSizeAcc = 0;

    // ====================================================================================================================
    
    public WorkerTransformer(int workerId, long t0, AtomicLong t1) {

        this.workerId = workerId;
        this.t0 = t0;
        this.t1 = t1;

        this.ws = WorkerStatic.get(workerId);
        this.logger = CustomLogger.getWorkerInstance(workerId);

        logger.log(taskInstance + ", Worker " + workerId + " WorkerTransformer started");

        this.pBestWeights = Dl4jParamUtils.modelToFlatList(ws.model);
        this.velocity =  new float[this.pBestWeights.length];

        if(FULLY_INFORMED == true) {
            stateStoreName = "pBestStore";
            keyName = "pBest" + workerId;   // this is the unique key, necessary for the statestore to work
        } else {
            stateStoreName = "gBestStore";
            keyName = "gBest";              // the gBest is only one at a time, we only need 1 key (gBest weights get constantly overwritten)
                                            // gBest is not per worker, its globally for all workers
        }

        this.control = CoordinatorControl.getInstance();
    }

    //=========================================================================================================================

    @Override
    @SuppressWarnings("unchecked")
    public void init(ProcessorContext context) {
        this.context = context;
        this.bestStore = (ReadOnlyKeyValueStore<String, ValueAndTimestamp<WeightsMessage>>) context.getStateStore(stateStoreName);
            // bestStore is used for both FULLY_INFORMED and Neighborhood Best

        context.schedule(Duration.ofMillis(CHECK_EVERY_MS), PunctuationType.WALL_CLOCK_TIME, timestamp -> {

            long sinceStartNs = System.nanoTime() - t0;
            if (sinceStartNs < TimeUnit.MILLISECONDS.toNanos(IDLE_GRACE_MS)) {
                return; // don't check yet
            }

            long idleNs = System.nanoTime() - t1.get();
            if (idleNs >= TimeUnit.MILLISECONDS.toNanos(IDLE_MS)) {
                logger.log(taskInstance + ", [Worker " + workerId + "] Idle for " + (idleNs / 1_000_000) + " ms -> requesting stop");
                CoordinatorControl.getInstance().requestStop(workerId);
            }
        });

    }

    //=========================================================================================================================

    @Override
    public KeyValue<String, WeightsMessage> transform(String key, DataMessage value) {

        start = System.nanoTime();

        if (!printedOffset) {
            printedOffset = true;
            logger.log(taskInstance + ", Starting at -> " + "Offset: " + context.offset() + ", Partition: " + context.partition() +
                            ", Topic: " + context.topic());

            // logger.log(taskInstance + ", Sample DataMessage: " + value.toString());
            logger.log(taskInstance + ", Sample DataMessage: " + value.toStringFull());
            
            seenPartitions.add(context.partition());
        }

        lastOffset = context.offset();

        if (value == null) {
            return null;
        }

        buffer.add(value);

        if (buffer.size() < TRAIN_SIZE) {   // if not completed the batch, just return
            return null;                    // bufferSize is always: 100.0
        }

        bufferSizeAcc += buffer.size();
        ws.stats.reset();
        logger.log("AFTER reset: pBestLoss=" + ws.stats.getPBestLoss()
                + " lastSentPBestLoss=" + ws.stats.getLastSentPBestLoss());

        float[] accLoss = ws.predictor.callPredictionsBatch(buffer);    // this is a forward pass
        if(accLoss == null) {
            control.requestStopFinal(); // a serious error has happend
            return null;
        }
        accuracy = accLoss[0];
        loss = accLoss[1];
        nSamples = (int) accLoss[2];
        nCorrect = (int) accLoss[3];
        forwardPassNs += accLoss[4];
        countForwardPass += 1;                  // forward pass completed

        // logger.log(taskInstance + ", accuracy on current batch: " + accuracy + ", with nSamples: " + nSamples + " and nCorrect: " + nCorrect);

        buffer.clear();
        batchesRead++;

        if (loss == 0f) {
            System.out.println("Loss Invalid");
            return null;
        }

        float[] weights = Dl4jParamUtils.modelToFlatList(ws.model);

        // =================================================================================================
        // Send pBest or current weights ===================================================================

        boolean improvement_to_pBest = (Math.round(loss * 1000f) / 1000f) < ws.stats.getPBestLoss(); 
                // boolean has pBest changed (improved) or not ?
        
        if(improvement_to_pBest) {    // update self always when improvement 

            ws.stats.setPBestAccuracy(accuracy);
            ws.stats.setPBestLoss(loss);

            this.pBestWeights = weights;
            logger.log(taskInstance + ", Improved pBest with loss: " + ws.stats.getPBestLoss() + " and accuracy: " + ws.stats.getBestAccuracy()
                        + ", with weights: " + Dl4jParamUtils.sampleFlat(weights, SAMPLING_CONSTANT));
        }

        // =========================================================================================================
        // Update to next position, Using the State Store ===================================================================

        if (FULLY_INFORMED == true) {

            List<float[]> pBestList = readPBestStore();

            if (pBestList == null || pBestList.isEmpty()) {     // if no neighbor has yet reported their pBest
                logger.log(taskInstance + ", No neighbor pBest found; skipping social update this round.");
                velocity = ws.psoUpdater.updateX(ws.model, null, accuracy);

            } else {
                // logger.log(taskInstance + ", pBest Weights: \n" + Dl4jParamUtils.sampleFlats(pBestList));
                velocity = ws.psoUpdater.updateX(ws.model, pBestList, accuracy);
                // velocity = ws.psoUpdater.updateXAdaptive(ws.model, pBestList, accuracy);
                
            }

        } else {

            float[] gBestWeights = readGBestStore();

            if (gBestWeights == null) {
                gBestWeights = new float[this.pBestWeights.length];
                velocity = ws.psoUpdater.updateX(ws.model, this.pBestWeights, gBestWeights, accuracy); // social term is ignored effectevly. 
            } else {
                logger.log(taskInstance + ", gBest Weights: " + Dl4jParamUtils.sampleFlat(gBestWeights, SAMPLING_CONSTANT) + 
                    ", gBest Accuracy: " + ws.local_gBestAccuracy + ", lastActivitySeconds: " + lastActivitySeconds);
                velocity = ws.psoUpdater.updateX(ws.model, this.pBestWeights, gBestWeights, accuracy);

            }
        }

        // =================================================================================================
        // Send pBest or current weights ===================================================================

        // Filtering: is the loss significant enough to be reported ?
        boolean significant_diff = true;

        if(FILTER_ENABLED == true) {
            if(FULLY_INFORMED == true) {

                significant_diff = Math.abs(loss - ws.stats.getLastSentPBestLoss()) / (Math.abs(ws.stats.getLastSentPBestLoss()) + eps) > SIGNIFICANT_LOSS_DIFF;
                    // in comparison to the last pBest of a worker, dont send if insignificant, other workers already have a good enough version
            
            } else {

                significant_diff = Math.abs(loss - ws.local_gBestLoss) / (Math.abs(ws.local_gBestLoss) + eps) > 0.3 * SIGNIFICANT_LOSS_DIFF;
                    // in comparison to the last global model, dont send if insignificant, other workers already have a good enough version of the global model
                    // this is a much more damaging filter, because the global affects all workers as the only sense of direction
                    // thats why 0.3 
            }
            logger.log("Filtering takes place: " + (significant_diff == false) + ", since significance is: " + significant_diff);
        }
        
        if(significant_diff && improvement_to_pBest) {    // update self always when improvement 

            ws.stats.setLastSentPBestLoss(loss);

            String msgIndex = java.util.UUID.randomUUID().toString();

            logger.log(taskInstance + ", Improved and sending pBest with loss: " + ws.stats.getPBestLoss() + " and accuracy: " + ws.stats.getBestAccuracy()
                    + ", msgIndex = " + msgIndex + ", with weights: " + Dl4jParamUtils.sampleFlat(weights, SAMPLING_CONSTANT));

            WeightsMessage msg = new WeightsMessage(workerId, msgIndex, accuracy, loss, weights);

            return new KeyValue<>(keyName, msg);
        }

        // =========================================================================================================
        // Send current position after N_BATCHES, for FedAvg + Swarm Monitoring. Reset batchesRead

        if (batchesRead >= N_BATCHES) {   

            logger.log(taskInstance + ", Sending current weights ...");

            batchesRead = 0;
            String msgIndex = java.util.UUID.randomUUID().toString();

            WeightsMessage msg = new WeightsMessage(workerId, msgIndex, accuracy, loss, weights);

            return new KeyValue<>("current_weights", msg);
        }

        // =========================================================================================================
        // Logging and Time

        updateTime();   // is updated  every time a new buffer has been processed
        
        logger.log(taskInstance + ", Time: " + lastActivitySeconds + ", with accuracy: " + accuracy +
                ", with loss: " + loss + ", with velocity (magnitude): " + Dl4jParamUtils.rmsScaled(velocity, 100) + 
                ", updated Model to: " + Dl4jParamUtils.sampleFlat(Dl4jParamUtils.modelToFlatList(ws.model), SAMPLING_CONSTANT) +
                ", with Velocities: " + Dl4jParamUtils.sampleFlat(velocity, SAMPLING_CONSTANT)
        );  // * 100 is for the user, just scale it upwards 
                
        end = System.nanoTime();
        sumElapsedNs += (end - start);  // most of the time all we are measuring is the average time of forward pass (from callPredictions). 
                                        // Doesnt trigger when we are collecting a batch
        count++;

        return null;
    }

    //=========================================================================================================================

    private void updateTime() {
        long now = System.nanoTime();
        t1.set(now);
        lastActivitySeconds = Math.round(((now - t0) / 1_000_000_000.0) * 1000.0) / 1000.0;
    }

    //=========================================================================================================================

    private List<float[]> readPBestStore() {     // for FULLY_INFORMED bestStore

        List<float[]> neighbors = new ArrayList<>();

        if (bestStore == null) {
            logger.log(taskInstance + ", readPBestStore: bestStore is null");
            return neighbors;
        }
        
        logger.log(taskInstance + ", pBest Weights: ");

        try (KeyValueIterator<String, ValueAndTimestamp<WeightsMessage>> it = bestStore.all()) {
                                                                // this is GlobalKTable it will run for all of them
            while (it.hasNext()) {  // iterate on every Statestore (they come from different workers)
                                    // They have names: "pBest" + workerId


                KeyValue<String, ValueAndTimestamp<WeightsMessage>> entry = it.next();
                WeightsMessage msg = entry.value.value(); 
                if (msg == null || msg.weights == null || msg.weights.length == 0) {
                    continue;
                }

                float[] pBestArr = msg.weights;

                neighbors.add(pBestArr);

                logger.log(msg.workerId + ")" + Dl4jParamUtils.sampleFlat(pBestArr, SAMPLING_CONSTANT) + ", with accuracy = " + msg.accuracy + 
                        ", with loss = " + msg.loss + ", with msgIndex: " + msg.msgIndex);
            }


        } catch (Exception e) {
            logger.log(taskInstance + ", Error iterating bestStore: " + e.getMessage());
            e.printStackTrace();
        }

        return neighbors;
    }


    //=========================================================================================================================

    private float[] readGBestStore() {

        if (bestStore == null) {
            logger.log(taskInstance + ", bestStore is null");
            return null;
        }

        ValueAndTimestamp<WeightsMessage> wrapper = bestStore.get(keyName);
        if (wrapper == null) {
            logger.log(taskInstance + ", gBestWeights returned null (no entry for key '" + keyName + "')");
            return null;
        }

        WeightsMessage best = wrapper.value();
        if (best == null || best.weights == null || best.weights.length == 0) {
            logger.log(taskInstance + ", gBestWeights is empty for key '" + keyName + "'");
            return null;
        }

        float[] gBestWeights = best.weights;
        
        if(best.loss < ws.local_gBestLoss) {  // update ws.local_gBestAccuracy
            ws.local_gBestLoss = best.loss;
            ws.local_gBestAccuracy = best.accuracy;
        }

        if (gBestWeights == null || gBestWeights.length == 0) {
            logger.log(taskInstance + ", gBestWeights is empty for key '" + keyName + "'");
            return null;
        }

        if (best.loss < ws.stats.getLastSeenGBestLoss() - 1e-9) {
            ws.stats.setLastSeenGBestLoss(best.loss);
            logger.log(taskInstance + ", gBest updated ! , with loss = " + best.loss + " and acc = " + best.accuracy);
        }

        return gBestWeights;
    }

    // =====================================================================================================================

    // private void dumpBestStore() {

    //     if (bestStore == null) {
    //         logger.log(taskInstance + ", [bestStore] Store is null!");
    //         return;
    //     }

    //     try (KeyValueIterator<String, ValueAndTimestamp<WeightsMessage>> it = bestStore.all()) {

    //         boolean empty = true;

    //         while (it.hasNext()) {
    //             empty = false;
    //             KeyValue<String, ValueAndTimestamp<WeightsMessage>> entry = it.next();

    //             WeightsMessage msg = entry.value.value(); // unwrap
    //             if (msg == null) {
    //                 logger.log(taskInstance + ", [bestStore] key = " + entry.key + ", value = null");
    //                 continue;
    //             }

    //             int nWeights = (msg.weights != null) ? msg.weights.length : 0;

    //             logger.log(taskInstance + " " +
    //                 "[bestStore] key = " + entry.key +
    //                 ", id_worker = " + msg.workerId +
    //                 ", msgIndex = " + msg.msgIndex +
    //                 ", accuracy = " + msg.accuracy +
    //                 ", loss = " + msg.loss +
    //                 ", nWeights = " + nWeights
    //             );
    //         }

    //         if (empty) {
    //             logger.log(taskInstance + ", [bestStore] Store is empty!");
    //         }

    //     } catch (Exception e) {
    //         logger.log(taskInstance + ", Error while dumping bestStore: " + e.getMessage());
    //         e.printStackTrace();
    //     }
    // }



    //=========================================================================================================================
    
    private double normL2PerDim(float[] a, float[] b) { // computes L2 Distance, normalized
        // dist = ||x − center||₂ / √d, dist = 0.01 means each parameter / weight differs by ~0.01 on average
        // Typical magnitudes after training: 0.1 – 1.0 and small changes are considered to be: 0.001 - 0.01
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = (double)a[i] - (double)b[i];
            sum += d * d;
        }
        return Math.sqrt(sum) / Math.sqrt(a.length);
    }
    
    //=========================================================================================================================

    private float[] computeMeanPBestFromStore() {

        if (bestStore == null) {
            logger.log(taskInstance + ", [Convergence] bestStore is null");
            return null;
        }

        double[] sum = null;
        int n = 0;

        try (KeyValueIterator<String, ValueAndTimestamp<WeightsMessage>> it = bestStore.all()) {
            while (it.hasNext()) {
                KeyValue<String, ValueAndTimestamp<WeightsMessage>> entry = it.next();
                WeightsMessage msg = entry.value.value();
                if (msg == null || msg.weights == null || msg.weights.length == 0) continue;

                float[] w = msg.weights;

                if (sum == null) sum = new double[w.length];
                if (w.length != sum.length) {
                    logger.log(taskInstance + ", [Convergence] pBest length mismatch, skipping key=" + entry.key);
                    continue;
                }

                for (int i = 0; i < w.length; i++) sum[i] += w[i];
                n++;
            }
        } catch (Exception e) {
            logger.log(taskInstance + ", [Convergence] error while reading pBest store: " + e.getMessage());
            return null;
        }

        if (sum == null || n == 0) return null;

        float[] mean = new float[sum.length];
        for (int i = 0; i < sum.length; i++) mean[i] = (float)(sum[i] / n);
        return mean;
    }

    //=========================================================================================================================

    @Override
    public void close() {

        if (!buffer.isEmpty()) {
            buffer.clear();
        }

        logger.log(taskInstance + ", Seen partitions: " + seenPartitions + ", with lastOffset: " + lastOffset);

        double avgMs = (sumElapsedNs / 1_000_000.0) / count;    // this is the overall time of processing a batch
        double avgForwardPassMs = forwardPassNs / countForwardPass;     // this is just the forward pass part of it (1 batch => 1 forward pass)
                        // what we are observing is that forward pass takes the most amount of time inside the entire batch processing

        logger.log(taskInstance + ", average elapsed time per batch: " + String.format("%.3f ms", avgMs)
                + " over " + count + " batches" + ", average forwardPassMs: " + avgForwardPassMs);
        
        // Report on convergence: ==============================================================
        if(ws.printedReport == false) {
            float[] center = null;

            if (FULLY_INFORMED == false) {
                center = readGBestStore();
                if (center == null) {
                    logger.log(taskInstance + ", [Convergence] gBest not available -> cannot evaluate convergence.");
                    return;
                }
            } else {
                center = computeMeanPBestFromStore();
                if (center == null) {
                    logger.log(taskInstance + ", [Convergence] pBest mean not available -> cannot evaluate convergence.");
                    return;
                }
            }

            float[] x = Dl4jParamUtils.modelToFlatList(ws.model);
            double dist = normL2PerDim(x, center);      // dist ≈ 0.05 → each weight differs by ~0.05 on average
            double radius = CONVERGENCE_ALPHA * Dl4jParamUtils.rms(center); // RMS / typical magnitude of weights
                // The particle is converged if, on average, each weight differs from the center by 
                // less than CONVERGENCE_ALPHA * 100% (i.e. 10%) of a typical weight’s magnitude.

            boolean converged = dist <= radius;

            logger.log(taskInstance + " dist = " + String.format("%.4f", dist)
                    + " radius = " + Dl4jParamUtils.round((float) radius, 4)
                    + ", with velocity (magnitude): " + Dl4jParamUtils.rmsScaled(velocity, 100) 
                    + " => " + (converged ? "CONVERGED" : "NOT_CONVERGED"));

            System.out.println("Dist = " + String.format("%.4f", dist)
                    + ", Radius = " + Dl4jParamUtils.round((float) radius, 4)
                    + ", with velocity (magnitude): " + Dl4jParamUtils.rmsScaled(velocity, 100) 
                    + " => " + (converged ? "CONVERGED" : "NOT_CONVERGED"));
            
            logger.log("Average bufferSize: " + Dl4jParamUtils.round(bufferSizeAcc / countForwardPass, 2));
            
            ws.printedReport = true;
        }

    }
}

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
    private final float SIGNIFICANT_LOSS_DIFF = cfg.SIGNIFICANT_LOSS_DIFF;
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;

    private final CustomLogger logger;

    private ProcessorContext context;

    private ReadOnlyKeyValueStore<String, ValueAndTimestamp<WeightsMessage>> bestStore;

    private final List<DataMessage> buffer = new ArrayList<>();

    private final MultiLayerNetwork model;
    private final Stats stats;
    private final BatchPrediction predictor;
    private final PsoUpdater psoUpdater;

    private float[] pBestWeights;
    private int batchesRead = 0;

    private boolean printedOffset = false;

    private String stateStoreName;
    private String keyName;

    private boolean accuracy_invalid = false;
    private boolean loss_invalid = false;

    private float accuracy = -1f;
    private float loss = 10000f;
    private int nSamples = 0;
    private int nCorrect = 0;

    private float local_gBestAccuracy = -1f;
    private float local_gBestLoss = 10000f;

    private long t0;
    private final AtomicLong t1;
    private double lastActivitySeconds = 0.0;

    private final Set<Integer> seenPartitions = ConcurrentHashMap.newKeySet();
    private long lastOffset = 0;

    private static final long IDLE_MS = 3000; // <-- set what you want (e.g. 3s)
    private static final long CHECK_EVERY_MS = 250; // how often we check

    // ====================================================================================================================
    
    public WorkerTransformer(int workerId, long t0, AtomicLong t1) {

        this.workerId = workerId;
        this.t0 = t0;
        this.t1 = t1;

        this.logger = CustomLogger.getWorkerInstance(workerId);

        this.model = Dl4jModelFactory.createModel();
        this.stats = new Stats();
        this.predictor = new BatchPrediction(model, stats, logger);
        this.psoUpdater = new PsoUpdater(model, workerId);

        this.pBestWeights = Dl4jParamUtils.modelToFlatList(model);

        logger.log("Worker " + workerId + " WorkerTransformer started");

        if(FULLY_INFORMED == true) {
            stateStoreName = "pBestStore";
            keyName = "pBest" + workerId;
        } else {
            stateStoreName = "gBestStore";
            keyName = "gBest";
        }
    }

    @Override
    @SuppressWarnings("unchecked")
    public void init(ProcessorContext context) {
        this.context = context;
        this.bestStore = (ReadOnlyKeyValueStore<String, ValueAndTimestamp<WeightsMessage>>) context.getStateStore(stateStoreName);

        context.schedule(Duration.ofMillis(CHECK_EVERY_MS), PunctuationType.WALL_CLOCK_TIME, timestamp -> {
            long idleNs = System.nanoTime() - t1.get();
            if (idleNs >= TimeUnit.MILLISECONDS.toNanos(IDLE_MS)) {
                logger.log("[Worker " + workerId + "] Idle for " + (idleNs / 1_000_000) + " ms -> requesting stop");
                CoordinatorControl.getInstance().requestStop(workerId);
            }
        });

    }

    //=========================================================================================================================

    @Override
    public KeyValue<String, WeightsMessage> transform(String key, DataMessage value) {

        if (!printedOffset) {
            printedOffset = true;
            logger.log("Starting at -> " + "Offset: " + context.offset() + ", Partition: " + context.partition() +
                            ", Topic: " + context.topic());

            logger.log("Sample DataMessage: " + value.toString());
            seenPartitions.add(context.partition());
        }
        lastOffset = context.offset();

        if (value == null) {
            return null;
        }

        buffer.add(value);

        if (buffer.size() < TRAIN_SIZE) {
            return null;
        }

        stats.reset();

        try {
            float[] accLoss = predictor.callPredictionsBatch(buffer);
            accuracy = accLoss[0];
            loss = accLoss[1];
            nSamples = (int) accLoss[2];
            nCorrect = (int) accLoss[3];
        } catch (Exception e) {
            logger.log("Error iterating bestStore: " + e.getMessage());
            System.out.println("Error iterating bestStore: " + e.getMessage());
            e.printStackTrace();
        }

        buffer.clear();
        batchesRead++;

        if (loss == 0f) {
            if(loss_invalid == false) {
                System.out.println("Loss Invalid");
                loss_invalid = true;
            }
            System.out.println("Loss Invalid");
            return null;
        }

        accuracy_invalid = false;
        loss_invalid = false;

        float[] weights = Dl4jParamUtils.modelToFlatList(model);

        // Send pBest or current weights ===================================================================

        boolean improvement_to_pBest = (Math.round(loss * 1000f) / 1000f) < stats.getBestLoss(); 
                // boolean has pBest improved or not ?

        boolean significant_diff_to_gBest = Math.abs(loss - local_gBestLoss) > SIGNIFICANT_LOSS_DIFF;
                // is the loss significant enough to be reported ?

        if(improvement_to_pBest) {    // update self always when improvement 

            stats.setBestAccuracy(accuracy);
            stats.setBestLoss(loss);

            this.pBestWeights = weights;

            if(significant_diff_to_gBest) { // send only when significant improvement
            
                String msgIndex = java.util.UUID.randomUUID().toString();

                logger.log("Improved loss: " + stats.getBestLoss() + " and accuracy: " + stats.getBestAccuracy() +
                            ", actuall loss: " + loss + ", msgIndex = " + msgIndex +
                            ", nSamples: " + nSamples + ", nCorrect: " + nCorrect);

                WeightsMessage msg = new WeightsMessage(workerId, msgIndex, accuracy, loss, weights);

                return new KeyValue<>(keyName, msg);
            }
        }

        if (batchesRead >= N_BATCHES) {    // send current position after N_BATCHES. For FedAvg

            logger.log("Sending current weights ...");

            batchesRead = 0;
            String msgIndex = java.util.UUID.randomUUID().toString();

            WeightsMessage msg = new WeightsMessage(workerId, msgIndex, accuracy, loss, weights);

            return new KeyValue<>("current_weights", msg);
        }

        float[] velocity = new float[this.pBestWeights.length];


        // Update to next position, Get the State Store ===================================================================

        if (FULLY_INFORMED == true) {

            List<float[]> neighborPBestList = readNeighborPBestList();

            if (neighborPBestList == null || neighborPBestList.isEmpty()) {
                logger.log("No neighbor pBest found; skipping social update this round.");
                velocity = psoUpdater.updateX(model, null);

            } else {
                logger.log("pBest Weights with accuracy: \n" + Dl4jParamUtils.sampleFlats(neighborPBestList));
                velocity = psoUpdater.updateX(model, neighborPBestList);
            }

        } else {

            float[] gBestWeights = readBestWeights();
            if (gBestWeights == null) {
                gBestWeights = new float[this.pBestWeights.length];
            } else {
                logger.log("gBest Weight: " + Dl4jParamUtils.sampleFlat(gBestWeights, SAMPLING_CONSTANT) + 
                    ", gBest Accuracy: " + local_gBestAccuracy + ", lastActivitySeconds: " + lastActivitySeconds);
            }

            velocity = psoUpdater.updateX(model, this.pBestWeights, gBestWeights);
        }

        updateTime();

        logger.log(lastActivitySeconds + 
                ", updated Model to: " + Dl4jParamUtils.sampleFlat(Dl4jParamUtils.modelToFlatList(model), SAMPLING_CONSTANT) +
                ", with loss: " + loss + ", with velocity (magnitude): " + Dl4jParamUtils.magnitude(velocity) + 
                ", with accuracy: " + accuracy);

        return null;
    }

    //=========================================================================================================================

    private void updateTime() {
        long now = System.nanoTime();
        t1.set(now);
        lastActivitySeconds = Math.round(((now - t0) / 1_000_000_000.0) * 100.0) / 100.0;
    }

    //=========================================================================================================================

    private List<float[]> readNeighborPBestList() {

        List<float[]> neighbors = new ArrayList<>();

        if (bestStore == null) {
            logger.log("readNeighborPBestList: bestStore is null");
            return neighbors;
        }

        try (KeyValueIterator<String, ValueAndTimestamp<WeightsMessage>> it = bestStore.all()) {
                                                                // this is GlobalKTable it will run for all of them
            // int count = 0;
            while (it.hasNext()) {  // iterate on every Statestore (they come from different workers)
                                    // They have names: "pBest" + workerId

                // logger.log("Runnig: " + count);
                // count = count + 1;  
                                 
                KeyValue<String, ValueAndTimestamp<WeightsMessage>> entry = it.next();
                WeightsMessage msg = entry.value.value(); 
                if (msg == null || msg.weights == null || msg.weights.length == 0) {
                    continue;
                }

                float[] pBestArr = msg.weights;

                neighbors.add(pBestArr);
            }


        } catch (Exception e) {
            logger.log("Error iterating bestStore: " + e.getMessage());
            e.printStackTrace();
        }

        return neighbors;
    }


    //=========================================================================================================================

    private float[] readBestWeights() {

        if (bestStore == null) {
            logger.log("bestStore is null");
            return null;
        }

        ValueAndTimestamp<WeightsMessage> wrapper = bestStore.get(keyName);
        if (wrapper == null) {
            logger.log("gBestWeights returned null (no entry for key '" + keyName + "')");
            return null;
        }

        WeightsMessage best = wrapper.value();
        if (best == null || best.weights == null || best.weights.length == 0) {
            logger.log("gBestWeights is empty for key '" + keyName + "'");
            return null;
        }

        float[] gBestWeights = best.weights;
        // if(best.accuracy > local_gBestAccuracy) {  // update local_gBestAccuracy
        //     local_gBestAccuracy = best.accuracy;
        // }
        
        if(best.loss < local_gBestLoss) {  // update local_gBestAccuracy
            local_gBestLoss = best.loss;
            local_gBestAccuracy = best.accuracy;
        }

        if (gBestWeights == null || gBestWeights.length == 0) {
            logger.log("gBestWeights is empty for key '" + keyName + "'");
            return null;
        }

        return gBestWeights;
    }

    // =====================================================================================================================

    // private void dumpBestStore() {

    //     if (bestStore == null) {
    //         logger.log("[bestStore] Store is null!");
    //         return;
    //     }

    //     try (KeyValueIterator<String, ValueAndTimestamp<WeightsMessage>> it = bestStore.all()) {

    //         boolean empty = true;

    //         while (it.hasNext()) {
    //             empty = false;
    //             KeyValue<String, ValueAndTimestamp<WeightsMessage>> entry = it.next();

    //             WeightsMessage msg = entry.value.value(); // unwrap
    //             if (msg == null) {
    //                 logger.log("[bestStore] key = " + entry.key + ", value = null");
    //                 continue;
    //             }

    //             int nWeights = (msg.weights != null) ? msg.weights.length : 0;

    //             logger.log(
    //                 "[bestStore] key = " + entry.key +
    //                 ", id_worker = " + msg.idWorker +
    //                 ", msgIndex = " + msg.msgIndex +
    //                 ", accuracy = " + msg.accuracy +
    //                 ", loss = " + msg.loss +
    //                 ", nWeights = " + nWeights
    //             );
    //         }

    //         if (empty) {
    //             logger.log("[bestStore] Store is empty!");
    //         }

    //     } catch (Exception e) {
    //         logger.log("Error while dumping bestStore: " + e.getMessage());
    //         e.printStackTrace();
    //     }
    // }



    //=========================================================================================================================

    @Override
    public void close() {
        if (!buffer.isEmpty()) {
            buffer.clear();
        }

        logger.log("Seen partitions: " + seenPartitions + ", with lastOffset: " + lastOffset);

    }
}

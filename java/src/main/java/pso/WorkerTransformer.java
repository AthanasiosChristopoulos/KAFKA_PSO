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
    private final float SIGNIFICANT_LOSS_DIFF = cfg.SIGNIFICANT_LOSS_DIFF;
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;

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

    private boolean printedOffset = false;

    private String stateStoreName;
    private String keyName;

    private boolean accuracy_invalid = false;
    private boolean loss_invalid = false;

    private float accuracy = -1f;
    private float loss = 10000f;
    private int nSamples = 0;
    private int nCorrect = 0;

    private long t0;
    private final AtomicLong t1;
    private double lastActivitySeconds = 0.0;

    private final Set<Integer> seenPartitions = ConcurrentHashMap.newKeySet();
    private long lastOffset = 0;

    private static final long IDLE_MS = 3000; // <-- set what you want (e.g. 3s)
    private static final long CHECK_EVERY_MS = 250; // how often we check

    private static final AtomicInteger INSTANCE_SEQ = new AtomicInteger(0);
    private final int instanceNo = INSTANCE_SEQ.incrementAndGet();
    private final String taskInstance = instanceNo + "@" + Integer.toHexString(System.identityHashCode(this));
    private String taskTag = "task=UNKNOWN";
    private static boolean ONCE = false;

    // ====================================================================================================================
    
    public WorkerTransformer(int workerId, long t0, AtomicLong t1) {

        this.workerId = workerId;
        this.t0 = t0;
        this.t1 = t1;

        this.ws = WorkerStatic.get(workerId);
        this.logger = CustomLogger.getWorkerInstance(workerId);

        // logger.log(taskInstance + " " +taskInstance + " thread=" + Thread.currentThread().getName()+ ", Worker " + workerId + " WorkerTransformer started");
        logger.log(taskInstance + ", Worker " + workerId + " WorkerTransformer started");

        this.pBestWeights = Dl4jParamUtils.modelToFlatList(ws.model);

        if(FULLY_INFORMED == true) {
            stateStoreName = "pBestStore";
            keyName = "pBest" + workerId;   // this is the unique key, necessary for the statestore to work
        } else {
            stateStoreName = "gBestStore";
            keyName = "gBest";              // the gBest is only one at a time, we only need 1 key (gBest weights get constantly overwritten)
                                            // gBest is not per worker, its globally for all workers
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
                logger.log(taskInstance + ", [Worker " + workerId + "] Idle for " + (idleNs / 1_000_000) + " ms -> requesting stop");
                CoordinatorControl.getInstance().requestStop(workerId);
            }
        });

    }

    //=========================================================================================================================

    @Override
    public KeyValue<String, WeightsMessage> transform(String key, DataMessage value) {

        if (!printedOffset) {
            printedOffset = true;
            logger.log(taskInstance + ", Starting at -> " + "Offset: " + context.offset() + ", Partition: " + context.partition() +
                            ", Topic: " + context.topic());

            logger.log(taskInstance + ", Sample DataMessage: " + value.toString());
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

        ws.stats.reset();

        float[] accLoss = ws.predictor.callPredictionsBatch(buffer);
        accuracy = accLoss[0];
        loss = accLoss[1];
        nSamples = (int) accLoss[2];
        nCorrect = (int) accLoss[3];
        // logger.log(taskInstance + ", accuracy on current batch: " + accuracy + ", with nSamples: " + nSamples + " and nCorrect: " + nCorrect);

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

        float[] weights = Dl4jParamUtils.modelToFlatList(ws.model);

        // Send pBest or current weights ===================================================================

        boolean improvement_to_pBest = (Math.round(loss * 1000f) / 1000f) < ws.stats.getBestLoss(); 
                // boolean has pBest improved or not ?

        boolean significant_diff_to_gBest = Math.abs(loss - ws.local_gBestLoss) > SIGNIFICANT_LOSS_DIFF;
                // is the loss significant enough to be reported ?

        if(improvement_to_pBest) {    // update self always when improvement 

            ws.stats.setBestAccuracy(accuracy);
            // logger.log(taskInstance + " " +taskInstance + " thread = " + Thread.currentThread().getName()+ ", best Loss: " + ws.stats.getBestLoss());
            logger.log(taskInstance + ", best Loss: " + ws.stats.getBestLoss());
            ws.stats.setBestLoss(loss);

            this.pBestWeights = weights;

            if(significant_diff_to_gBest) { // send only when significant improvement
                
                String msgIndex = java.util.UUID.randomUUID().toString();

                logger.log(taskInstance + ", Improved loss: " + ws.stats.getBestLoss() + " and accuracy: " + ws.stats.getBestAccuracy() +
                            ", actuall loss: " + loss + ", msgIndex = " + msgIndex +
                            ", nSamples: " + nSamples + ", nCorrect: " + nCorrect);

                WeightsMessage msg = new WeightsMessage(workerId, msgIndex, accuracy, loss, weights);

                return new KeyValue<>(keyName, msg);
            }
        }

        if (batchesRead >= N_BATCHES) {    // send current position after N_BATCHES. For FedAvg

            logger.log(taskInstance + ", Sending current weights ...");

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
                logger.log(taskInstance + ", No neighbor pBest found; skipping social update this round.");
                velocity = ws.psoUpdater.updateX(ws.model, null);

            } else {
                logger.log(taskInstance + ", pBest Weights with accuracy: \n" + Dl4jParamUtils.sampleFlats(neighborPBestList));
                velocity = ws.psoUpdater.updateX(ws.model, neighborPBestList);
            }

        } else {

            float[] gBestWeights = readBestWeights();
            if (gBestWeights == null) {
                gBestWeights = new float[this.pBestWeights.length];
            } else {
                logger.log(taskInstance + ", gBest Weight: " + Dl4jParamUtils.sampleFlat(gBestWeights, SAMPLING_CONSTANT) + 
                    ", gBest Accuracy: " + ws.local_gBestAccuracy + ", lastActivitySeconds: " + lastActivitySeconds);
            }

            velocity = ws.psoUpdater.updateX(ws.model, this.pBestWeights, gBestWeights);
        }

        updateTime();

        logger.log(taskInstance + ", Time: " + lastActivitySeconds + 
                ", updated Model to: " + Dl4jParamUtils.sampleFlat(Dl4jParamUtils.modelToFlatList(ws.model), SAMPLING_CONSTANT) +
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
            logger.log(taskInstance + ", readNeighborPBestList: bestStore is null");
            return neighbors;
        }

        try (KeyValueIterator<String, ValueAndTimestamp<WeightsMessage>> it = bestStore.all()) {
                                                                // this is GlobalKTable it will run for all of them
            // int count = 0;
            while (it.hasNext()) {  // iterate on every Statestore (they come from different workers)
                                    // They have names: "pBest" + workerId

                // logger.log(taskInstance + ", Runnig: " + count);
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
            logger.log(taskInstance + ", Error iterating bestStore: " + e.getMessage());
            e.printStackTrace();
        }

        return neighbors;
    }


    //=========================================================================================================================

    private float[] readBestWeights() {

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
        // if(best.accuracy > ws.local_gBestAccuracy) {  // update ws.local_gBestAccuracy
        //     ws.local_gBestAccuracy = best.accuracy;
        // }
        
        if(best.loss < ws.local_gBestLoss) {  // update ws.local_gBestAccuracy
            ws.local_gBestLoss = best.loss;
            ws.local_gBestAccuracy = best.accuracy;
        }

        if (gBestWeights == null || gBestWeights.length == 0) {
            logger.log(taskInstance + ", gBestWeights is empty for key '" + keyName + "'");
            return null;
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
    //                 ", id_worker = " + msg.idWorker +
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

    @Override
    public void close() {
        if (!buffer.isEmpty()) {
            buffer.clear();
        }

        logger.log(taskInstance + ", Seen partitions: " + seenPartitions + ", with lastOffset: " + lastOffset);

    }
}

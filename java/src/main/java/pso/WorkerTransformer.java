package pso;

import org.apache.kafka.streams.kstream.Transformer;
import org.apache.kafka.streams.KeyValue;
import org.apache.kafka.streams.processor.ProcessorContext;
import org.apache.kafka.streams.processor.PunctuationType;
import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.streams.state.ReadOnlyKeyValueStore;
import org.apache.kafka.streams.state.ValueAndTimestamp;
import org.bytedeco.opencv.opencv_core.Size;

import dl4j_models.Dl4jParamUtils;

import org.apache.kafka.streams.state.KeyValueIterator;

import java.util.*;

import utils.*;
import state.*;
import message.data_message.*; 
import message.weights_message.*;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import java.time.Duration;

public class WorkerTransformer implements Transformer<String, DataMessage, KeyValue<String, WeightsMessage>> {

    private final int workerId;

    private static Config cfg = Config.getInstance();
    private final int BATCH_SIZE = cfg.BATCH_SIZE;
    private final int N_BATCHES = cfg.N_BATCHES;  
    private final int MONITORING_THRESHOLD_MAX = cfg.MONITORING_THRESHOLD_MAX;
    private final int MONITORING_THRESHOLD_MIN = cfg.MONITORING_THRESHOLD_MIN; 
    private int monitoring_threshold = cfg.N_BATCHES;
    private final boolean FULLY_INFORMED = cfg.FULLY_INFORMED;
    private final boolean FILTER_ENABLED = cfg.FILTER_ENABLED;
    private final float SIGNIFICANT_LOSS_DIFF = cfg.SIGNIFICANT_LOSS_DIFF;
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;
    private final float CONVERGENCE_ALPHA = cfg.CONVERGENCE_ALPHA;
    public final boolean ENABLE_NEIGHBORHOODS = cfg.ENABLE_NEIGHBORHOODS;
    public final int NEIGHBORHOOD_SIZE = cfg.NEIGHBORHOOD_SIZE; 
    public final boolean INCLUDE_SELF = cfg.INCLUDE_SELF;
    public final String NEIGHBORHOOD_TOPOLOGY = cfg.NEIGHBORHOOD_TOPOLOGY;
    public final boolean STOP_ON_CONVERGENCE = cfg.STOP_ON_CONVERGENCE;
    private final float CONVERGENCE_STRICTNESS_FACTOR = cfg.CONVERGENCE_STRICTNESS_FACTOR;

    private ProcessorContext context;

    private ReadOnlyKeyValueStore<String, ValueAndTimestamp<WeightsMessage>> bestStore;

    private final List<DataMessage> buffer = new ArrayList<>();

    private CustomLogger logger;
    private final WorkerStatic ws;
        
    private boolean printedOffset = false;

    private String stateStoreName;
    private String keyName;

    private float accuracy = -1f;
    private float loss = 10000f;
    private int nSamples = 0;
    private int nCorrect = 0;

    private long t0;
    private AtomicLong t_actually_started;
    private final AtomicLong t1;
    private double lastActivitySeconds = 0.0;
    
    // Performance Measurements ==============================================
    private long start = System.nanoTime();
    private long sumElapsedNs = 0;
    private int per_task_count = 0;

    private long startPredict = System.nanoTime();
    private long sumElapsedNsPredict = 0;

    private long startUpdateX = System.nanoTime();
    private long sumElapsedNsUpdateX = 0;

    private float forwardPassNs = 0;
    private int countForwardPass = 0;
    private static int countForwardPassesStatic = 0;

    // =======================================================================

    public KeyValue<String, WeightsMessage> out = null;

    private final Set<Integer> seenPartitions = ConcurrentHashMap.newKeySet();
    private long lastOffset = 0;
    private double eps = 1e-12;

    private static final long IDLE_MS = cfg.IDLE_MS; 
    private static final long CHECK_EVERY_MS = 100; 
    private static final long IDLE_GRACE_MS = 8000;

    private static final AtomicInteger INSTANCE_SEQ = new AtomicInteger(0);
    private final int instanceNo = INSTANCE_SEQ.incrementAndGet();
    private final String taskInstance = instanceNo + "@" + Integer.toHexString(System.identityHashCode(this));

    private final CoordinatorControl control; 

    private int bufferSizeAcc = 0;

    private int[] neighborIds;                 // precomputed ids
    private String[] neighborKeys;             // precomputed keys "pBestX"
    private int ringRadius;                    // NEIGHBORHOOD_SIZE/2
    private final float LOSS_INIT = 1e30f;

    private int consecutiveConvergence = 0;
    private static final int CONSECUTIVE_CONVERGENCE_REQUIRED = 6;

    private volatile WeightsMessage pendingPBestMsg = null;
    private long lastPBestForwardMs = 0;
    private long openedDebounceWindow = 0;
    private boolean pBestUpdatePending = false;

    private int MAX_UPDATES; // expected max updates (for clamping)
    private float TAU; 
    private final float LOSS_THRESHOLD_MAX = cfg.LOSS_THRESHOLD_MAX;           // e.g. 0.10f (10%)
    private final float LOSS_THRESHOLD_MIN = cfg.LOSS_THRESHOLD_MIN;     // e.g. 0.005f (0.5%)
    private float loss_threshold;

    // ====================================================================================================================
    
    public WorkerTransformer(int workerId, long t0, AtomicLong t_actually_started, AtomicLong t1, WorkerStatic ws) {
        this.logger = CustomLogger.getWorkerInstance(workerId);

        this.workerId = workerId;
        this.t0 = t0;
        this.t_actually_started = t_actually_started;
        this.t1 = t1;

        this.ws = ws;
        ws.numberOfTasks += 1;

        if (logger.isEnabled(0)) logger.log(taskInstance + ", Worker " + workerId + 
        " WorkerTransformer started");

        if(ENABLE_NEIGHBORHOODS == true || FULLY_INFORMED == true) {
            stateStoreName = "pBestStore";
            keyName = "pBest" + workerId;   // this is the unique key, necessary for the statestore to work
        } else {
            stateStoreName = "gBestStore";
            keyName = "gBest";              // the gBest is only one at a time, we only need 1 key (gBest weights get constantly overwritten)
                                            // gBest is not per worker, its globally for all workers
        }

        this.control = CoordinatorControl.getInstance();
    
        this.ringRadius = Math.max(0, NEIGHBORHOOD_SIZE / 2);

        // Build neighbor list only if neighborhoods enabled
        if (ENABLE_NEIGHBORHOODS) {
            this.neighborIds = computeNeighborIds(workerId, cfg.N_WORKERS, ringRadius, INCLUDE_SELF, NEIGHBORHOOD_TOPOLOGY);
            if(logger.isEnabled(2)) logger.log("neighborIds: " + Arrays.toString(neighborIds)); 
            this.neighborKeys = new String[neighborIds.length];
            for (int i = 0; i < neighborIds.length; i++) {
                neighborKeys[i] = "pBest" + neighborIds[i];     // if wieghtId - key isnt there then pBest weight gets filtered out
            }
        } else {
            this.neighborIds = null;
            this.neighborKeys = null;
        }

        logger.log("taskInstance: " + taskInstance + ", Thread.currentThread().getName(): " + Thread.currentThread().getName());

        if(FILTER_ENABLED) {
        
            if(cfg.INDEPENDENT_DATA_PROCESSING == true) {
                MAX_UPDATES = cfg.NUM_SAMPLES / BATCH_SIZE;
                
            } else {

                MAX_UPDATES = cfg.NUM_SAMPLES / (cfg.N_WORKERS * BATCH_SIZE);
            }

            double epsEnd = 1e-3;
            double diff = (double) LOSS_THRESHOLD_MAX - (double) LOSS_THRESHOLD_MIN;
            double tau = (double) MAX_UPDATES / Math.log(diff / epsEnd);
            logger.log("tau = " + tau + ", diff = " + diff);

            if (Double.isNaN(tau) || Double.isInfinite(tau) || tau < 1.0) tau = 1.0;    // so its valid

            this.TAU = (float) tau;

            logger.log("MAX_UPDATES = " + MAX_UPDATES + ", TAU = " + TAU);

            this.ws.predictorRefPsoVelocityMonitoring = new float[ws.flatModel.length];
        }
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
                return; 
            }

            long idleNs = System.nanoTime() - t1.get();
            
            if (idleNs >= TimeUnit.MILLISECONDS.toNanos(IDLE_MS) && !ws.endedWorker) {
                if (logger.isEnabled(2)) logger.log(taskInstance + 
                    ", Closed, because of idleness for " + (idleNs / 1_000_000) + " ms");
                System.out.println("[Worker" + workerId +"] Closed, because of idleness for " + (idleNs / 1_000_000) + " ms");
                CoordinatorControl.getInstance().requestStop(workerId);
                ws.endedWorker = true;
            }
        });

        // if(FILTER_ENABLED) {
        //     context.schedule(Duration.ofMillis(20), PunctuationType.WALL_CLOCK_TIME, ts -> {
        //         flushPendingPBest();
        //     });
        // }
    }
    //=========================================================================================================================
    // Prediction Model Stuff
    //=========================================================================================================================

    private double rmsDiff(float[] a, float[] b) {
        if (a == null || b == null || a.length != b.length) return Double.NaN;

        double sumSq = 0.0;
        for (int i = 0; i < a.length; i++) {
            double d = (double) a[i] - (double) b[i];
            sumSq += d * d;
        }
        return Math.sqrt(sumSq / a.length);
    }
    //=========================================================================================================================

    private float[] linearGrowthPredict(float[] ref, long t, long ts) {
        float[] pred = new float[ref.length];

        if (ts <= 0) {
            System.arraycopy(ref, 0, pred, 0, ref.length);
            return pred;
        }

        double scale = (double) t / (double) ts;
        for (int i = 0; i < ref.length; i++) {
            pred[i] = (float) (scale * ref[i]);
        }
        return pred;
    }
    //=========================================================================================================================

    private float[] estimateVelocity(float[] newer, long newerMs, float[] older, long olderMs) {
        float[] vel = new float[newer.length];
        long dtMs = Math.max(1L, newerMs - olderMs);

        for (int i = 0; i < newer.length; i++) {
            vel[i] = (float) (((double) newer[i] - (double) older[i]) / (double) dtMs);
        }
        return vel;
    }

    //=========================================================================================================================

    private float[] estimateAcceleration(float[] velNew, long tNew, float[] velOld, long tOld) {
        float[] acc = new float[velNew.length];
        long dtMs = Math.max(1L, tNew - tOld);

        for (int i = 0; i < velNew.length; i++) {
            acc[i] = (float) (((double) velNew[i] - (double) velOld[i]) / (double) dtMs);
        }
        return acc;
    }

    //=========================================================================================================================

    private float[] velocityAccelerationPredict(float[] refWeights, long refTimeMs, float[] vel, float[] acc, long nowMs) {

        float[] pred = new float[refWeights.length];
        long dt = Math.max(0L, nowMs - refTimeMs);

        for (int i = 0; i < refWeights.length; i++) {
            pred[i] = (float) (
                (double) refWeights[i]
                + (double) dt * (double) vel[i]
                + (double) dt * (double) dt * (double) acc[i]
            );
        }

        return pred;
    }

    //=========================================================================================================================

    private float[] observedVelocityPredict(float[] refWeights, long refTimeMs, float[] vel,long nowMs) {

        float[] pred = new float[refWeights.length];
        long dt = Math.max(0L, nowMs - refTimeMs);

        for (int i = 0; i < refWeights.length; i++) {
            pred[i] = (float) (
                (double) refWeights[i] + (double) dt * (double) vel[i]
            );
        }

        return pred;
    }
    //=========================================================================================================================

    private long currentSimulationTimeMs() {
        long now = java.time.Instant.now().toEpochMilli();
        // long start = Simulation.getSimulationStartMs();

        long simT = now - ws.simulationStartMs;
        return Math.max(simT, 1L);
    }

    //=========================================================================================================================

    private float[] psoVelocityPredict(float[] refWeights, float[] refVelocity, long t, long ts) {

        float[] pred = new float[refWeights.length];

        if (refVelocity == null || refVelocity.length != refWeights.length) {
            System.arraycopy(refWeights, 0, pred, 0, refWeights.length);
            return pred;
        }

        long dt = Math.max(0L, t - ts);

        for (int i = 0; i < refWeights.length; i++) {
            pred[i] = (float) (refWeights[i] + dt * refVelocity[i]);
        }

        return pred;
    }

    //=========================================================================================================================

    private void compareStaticVsLinearForPBest(float[] currentPBest) {
        long t = Math.max(1L, (long) ws.countForwardPasses + 1L);
        // long t = currentSimulationTimeMs();
        
        if (!ws.predictorInitializedPBest) {
            ws.predictorRefWeightsPBest = Arrays.copyOf(currentPBest, currentPBest.length);
            ws.predictorTsPBest = t;
            ws.predictorInitializedPBest = true;

            if (logger.isEnabled(1)) {
                logger.log(taskInstance + ", PBEST predictor baseline initialized at t_s = " + ws.predictorTsPBest);
            }
            return;
        }

        float[] staticPred = ws.predictorRefWeightsPBest;
        float[] linearPred = linearGrowthPredict(ws.predictorRefWeightsPBest, t, ws.predictorTsPBest);

        double staticErr = rmsDiff(currentPBest, staticPred);
        double linearErr = rmsDiff(currentPBest, linearPred);

        PredictorComparisonRegistry.record(
            PredictorComparisonRegistry.Kind.PBEST,
            staticErr,
            linearErr
        );

        if (logger.isEnabled(1)) {
            logger.log(taskInstance +
                ", PBEST predictor compare: t = " + t +
                ", ts = " + ws.predictorTsPBest +
                ", staticErr = " + staticErr +
                ", linearErr = " + linearErr +
                ", winner = " + (linearErr < staticErr ? "LINEAR" : (staticErr < linearErr ? "STATIC" : "TIE")));
        }

        ws.predictorRefWeightsPBest = Arrays.copyOf(currentPBest, currentPBest.length);
        ws.predictorTsPBest = t;
    }

    //=========================================================================================================================

    private void compareStaticVsLinearForMonitoring(float[] currentWeights) {
        long t = Math.max(1L, (long) ws.countForwardPasses + 1L);
        // long t = currentSimulationTimeMs();

        // current snapshot for future history update
        TimedWeightsSnapshot currentSnap = new TimedWeightsSnapshot(Arrays.copyOf(currentWeights, currentWeights.length), t);

        // first ever monitoring snapshot -> only initialize
        if (!ws.predictorInitializedMonitoring) {
            ws.predictorRefWeightsMonitoring = Arrays.copyOf(currentWeights, currentWeights.length);
            ws.predictorTsMonitoring = t;
            ws.predictorInitializedMonitoring = true;
            ws.predictorRefPsoVelocityMonitoring = (ws.velocity == null)
                ? null
                : Arrays.copyOf(ws.velocity, ws.velocity.length);

            ws.monPrev1 = currentSnap;

            if (logger.isEnabled(1)) {
                logger.log(taskInstance + ", MONITORING predictor baseline initialized at t_s = " + ws.predictorTsMonitoring);
            }
            return;
        }

        float[] staticPred = ws.predictorRefWeightsMonitoring;
        float[] linearPred = linearGrowthPredict(ws.predictorRefWeightsMonitoring, t, ws.predictorTsMonitoring);
        float[] psoVelPred = psoVelocityPredict(ws.predictorRefWeightsMonitoring, ws.predictorRefPsoVelocityMonitoring, t, ws.predictorTsMonitoring);

        double staticErr = rmsDiff(currentWeights, staticPred);
        double linearErr = rmsDiff(currentWeights, linearPred);
        double psoVelErr = rmsDiff(currentWeights, psoVelPred);

        // defaults when not enough history exists
        double observedVelErr = Double.POSITIVE_INFINITY;
        double vaErr = Double.POSITIVE_INFINITY;

        if (ws.monPrev1 != null && ws.monPrev2 != null) {
            logger.log("ws.monPrev1 and ws.monPrev2 arent null");

            float[] velObserved = estimateVelocity(ws.monPrev1.weights, ws.monPrev1.timeMs, ws.monPrev2.weights, ws.monPrev2.timeMs);
            float[] observedVelPred = observedVelocityPredict(ws.monPrev1.weights, ws.monPrev1.timeMs, velObserved, t);

            observedVelErr = rmsDiff(currentWeights, observedVelPred);

            if (ws.monPrev3 != null) {
                float[] velPrev = estimateVelocity(ws.monPrev2.weights, ws.monPrev2.timeMs, ws.monPrev3.weights, ws.monPrev3.timeMs);

                float[] accObserved = estimateAcceleration(velObserved, ws.monPrev1.timeMs, velPrev, ws.monPrev2.timeMs);

                float[] vaPred = velocityAccelerationPredict(ws.monPrev1.weights, ws.monPrev1.timeMs,velObserved, accObserved, t);

                vaErr = rmsDiff(currentWeights, vaPred);
            }
        } else {
            logger.log("ws.monPrev1 and ws.monPrev2 are null");
        }

        PredictorComparisonRegistry.recordMonitoring(
            staticErr,
            linearErr,
            psoVelErr,
            observedVelErr,
            vaErr
        );

        if (logger.isEnabled(1)) {
            logger.log(taskInstance +
                ", MONITORING predictor compare: t = " + t +
                ", ts = " + ws.predictorTsMonitoring +
                ", staticErr = " + staticErr +
                ", linearErr = " + linearErr +
                ", psoVelErr = " + psoVelErr +
                ", observedVelErr = " + observedVelErr +
                ", vaErr = " + vaErr);
        }

        // update baseline for static/linear/pso-velocity
        ws.predictorRefWeightsMonitoring = Arrays.copyOf(currentWeights, currentWeights.length);
        ws.predictorTsMonitoring = t;
        ws.predictorRefPsoVelocityMonitoring = (ws.velocity == null)
            ? null
            : Arrays.copyOf(ws.velocity, ws.velocity.length);

        // shift snapshot history for observed velocity / VA
        ws.monPrev3 = ws.monPrev2;
        ws.monPrev2 = ws.monPrev1;
        ws.monPrev1 = currentSnap;
    }

    //=========================================================================================================================
    //=========================================================================================================================
    //=========================================================================================================================

    @Override
    public KeyValue<String, WeightsMessage> transform(String key, DataMessage value) {
        
        if(CoordinatorControl.getInstance().isStopRequested(workerId)) {
            return null;
        }

        if(ws.firstActive == false) {
            t_actually_started.set(System.nanoTime());
            logger.log("t_actually_started: " + t_actually_started);
            logger.log("t0: " + t0);

            ws.simulationStartMs = java.time.Instant.now().toEpochMilli();
            
            ws.firstActive = true;
        }

        if (!printedOffset) {


            printedOffset = true;
            if (logger.isEnabled(2)) logger.log(taskInstance + ", Starting at -> " + 
                            "Offset: " + context.offset() + ", Partition: " + context.partition() +
                            ", Topic: " + context.topic());
            if (logger.isEnabled(2)) logger.log(taskInstance + 
                    ", Sample DataMessage: " + value.toStringFull());

            logger.log("Starting Delay: " + (System.nanoTime() - this.t0) / 1_000_000_000.0);

            seenPartitions.add(context.partition());    // if no records processed, this never runs
        }

        lastOffset = context.offset();

        if (value == null) {
            return null;
        }

        buffer.add(value);

        if (buffer.size() < BATCH_SIZE) {   // if not completed the batch, just return
            return null;                    // bufferSize is always: 100.0
        }
        out = null;

        start = System.nanoTime();
        startPredict = System.nanoTime();

        bufferSizeAcc += buffer.size();

        float[] accLoss = ws.predictor.callPredictionsBatch(buffer, ws.model, false);    // this is a forward pass
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

        sumElapsedNsPredict += (System.nanoTime() - startPredict);

        ws.batchesRead++;
        ws.incrementBatchesRead++;
        ws.incrementSamplesRead += buffer.size();

        buffer.clear();

        if (loss == 0f) {
            System.out.println("Loss Invalid");
            return null;
        }

        // =================================================================================================
        // Send pBest or current weights ===================================================================

        boolean improvement_to_pBest = (Math.round(loss * 1000f) / 1000f) < ws.stats.getPBestLoss(); 
                // boolean has pBest changed (improved) or not ?
        
        if(improvement_to_pBest) {    // update self always when improvement 

            ws.stats.setPBestAccuracy(accuracy);
            ws.stats.setPBestLoss(loss);

            for (int i = 0; i < ws.flatModel.length; i++) {
                ws.pBestWeights[i] = ws.flatModel[i];
            }

            ws.improved_pBest_count++;
            // compareStaticVsLinearForPBest(ws.pBestWeights);

            // if (logger.isEnabled(1)) logger.log(taskInstance + 
            //     ", Improved pBest with loss: " + ws.stats.getPBestLoss() + 
            //     " and accuracy: " + ws.stats.getBestAccuracy() +
            //     ", with weights: " + Dl4jParamUtils.sampleFlat(ws.flatModel, SAMPLING_CONSTANT));
        }

        // =========================================================================================================
        // Update to next position, Using the State Store ===================================================================

        startUpdateX = System.nanoTime();

        if (FULLY_INFORMED == true) {

            List<NeighborPBest> neighborList = readPBestStore();

            if (neighborList == null || neighborList.isEmpty()) {
                if (logger.isEnabled(2)) logger.log(taskInstance + 
                    ", No neighbor pBest found; skipping social update this round.");
                ws.velocity = ws.psoUpdater.updateX(null, accuracy, taskInstance);
            } else {
                ws.velocity = ws.psoUpdater.updateX(neighborList, accuracy, taskInstance);
            }

        } else {

            float[] gBestWeights = readGBestStore();

            if (gBestWeights == null) {     // gBestWeights not yet initialized
                gBestWeights = new float[ws.pBestWeights.length];
                ws.velocity = ws.psoUpdater.updateX(ws.pBestWeights, ws.pBestWeights, accuracy, taskInstance); // social term is ignored effectevly. 

            } else {

                if (logger.isEnabled(1)) logger.log(taskInstance + ", gBest Weights: " + 
                    Dl4jParamUtils.sampleFlat(gBestWeights, SAMPLING_CONSTANT) + ", gBest Accuracy: " 
                    + ws.local_gBestAccuracy + ", lastActivitySeconds: " + lastActivitySeconds);

                ws.velocity = ws.psoUpdater.updateX(ws.pBestWeights, gBestWeights, accuracy, taskInstance);

            }
        }

        sumElapsedNsUpdateX += (System.nanoTime() - startUpdateX);

        // =================================================================================================
        // Send pBest or current weights ===================================================================

        // Filtering: is the loss significant enough to be reported ?
        // we cant meassure performance from here ... this is just creating an object and returning it to the one that is going to send it.
        
        boolean significant_diff = true;
        updateThreshold(ws.countForwardPasses);

        if(FILTER_ENABLED == true) {

            if(FULLY_INFORMED == true) {

                significant_diff = Math.abs(loss - ws.stats.getLastSentPBestLoss()) / (Math.abs(ws.stats.getLastSentPBestLoss()) + eps) > loss_threshold;
                    // in comparison to the last pBest of a worker, dont send if insignificant, other workers already have a good enough version
            
            } else {

                significant_diff = Math.abs(loss - ws.local_gBestLoss) / (Math.abs(ws.local_gBestLoss) + eps) > 0.3 * loss_threshold;
                    // in comparison to the last global model, dont send if insignificant, other workers already have a good enough version of the global model
                    // this is a much more damaging filter, because the global affects all workers as the only sense of direction
                    // thats why 0.3 
            }
            if (logger.isEnabled(1)) logger.log("Filtering takes place: " + 
                (significant_diff == false) + 
                ", since significance is: " + significant_diff);
        }
        
        if(significant_diff && improvement_to_pBest) {    // update self always when improvement 

            ws.stats.setLastSentPBestLoss(loss);

            String msgIndex = java.util.UUID.randomUUID().toString();

            if (logger.isEnabled(1)) logger.log(taskInstance + 
                    ", Improved pBest with loss: " + ws.stats.getPBestLoss() + 
                    " and accuracy: " + ws.stats.getBestAccuracy() + ", msgIndex = " + msgIndex);
            
            ws.significant_pBest_count++;

            WeightsMessage msg = new WeightsMessage(workerId, msgIndex, accuracy, loss, ws.pBestWeights);

            if(FILTER_ENABLED) {
                ws.pBestCandidateCount++;
                pendingPBestMsg = msg;
                if(pBestUpdatePending == false) {
                    openedDebounceWindow = System.currentTimeMillis();
                    pBestUpdatePending = true;
                }

            } else {
                ws.incrementTotalMessagesSent("pBest");
                context.forward(keyName, msg);  // context.forward can be called 0 times, 1 time, or many times per input record.
            }   // This immediately pushes a record downstream from inside the processor.
        }

        // =========================================================================================================
        // Send current position after N_BATCHES, for FedAvg + Swarm Monitoring. Reset ws.batchesRead
        
        updateMonitoringThreshold(ws.countForwardPasses);
        boolean shouldSendMonitoring = (FILTER_ENABLED  && ws.batchesRead >= monitoring_threshold) || 
            (!FILTER_ENABLED && ws.batchesRead >= N_BATCHES);

        if (shouldSendMonitoring) {   // doesnt matter which partition sends localWeights message thats why ws.batchesRead 

            if (logger.isEnabled(1)) logger.log(taskInstance + 
                ", Sending current weights ...");

            ws.batchesRead = 0;
            String msgIndex = java.util.UUID.randomUUID().toString();

            float[] snapshot = Arrays.copyOf(ws.flatModel, ws.flatModel.length);    // The danger window for updating flatModel is before it becomes bytes.
            
            compareStaticVsLinearForMonitoring(snapshot);

            WeightsMessage msg = new WeightsMessage(workerId, msgIndex, accuracy, loss, snapshot);
            
            ws.incrementTotalMessagesSent("current_weights");
            out = new KeyValue<>("current_weights", msg);
        }

        // =========================================================================================================
        // Logging and Time
        logger.log("TOTAL_MESSAGES_SENT: " + ws.TOTAL_MESSAGES_SENT);
        
        updateTime();   // is updated  every time a new buffer has been processed. Need this for tracking lastActivity and overall activity time
        
        if (logger.isEnabled(0)) logger.log(taskInstance + 
                ", Time: " + lastActivitySeconds + ", with accuracy: " + accuracy +
                ", with loss: " + loss + ", with velocity (magnitude): " + Dl4jParamUtils.rmsScaled(ws.velocity, 100) + 
                ", updated Model to: " + Dl4jParamUtils.sampleFlat(ws.flatModel, SAMPLING_CONSTANT) +
                ", with Velocities: " + Dl4jParamUtils.sampleFlat(ws.velocity, SAMPLING_CONSTANT) +
                ", loss_threshold: " + loss_threshold + 
                ", monitoring_threshold: " + monitoring_threshold);  
        // * 100 is for the user, just scale it upwards 
                
        // =========================================================================================================
        // Check Convergence

        if(STOP_ON_CONVERGENCE && checkConvergence(true, false)) {
            consecutiveConvergence++;
            if(consecutiveConvergence >= CONSECUTIVE_CONVERGENCE_REQUIRED) {
                if (logger.isEnabled(2)) logger.log("Closed, because determined convergence");
                System.out.println("[Worker " + workerId + "] Closed, because determined convergence");
                CoordinatorControl.getInstance().requestStop(workerId);
            }
        } else {
            consecutiveConvergence = 0;
        }


        sumElapsedNs += (System.nanoTime() - start);  // most of the time all we are measuring is the average time of forward pass (from callPredictions). 
                                        // Doesnt trigger when we are collecting a batch
        per_task_count++; ws.countForwardPasses++; countForwardPassesStatic++;

        flushPendingPBest();    // this may send the actuall pBest
        
        // if(out != null) {
        //     System.out.println("[WorkerId " + workerId + "] Sending current weights topic");
        // }
        
        return out; // like this pBest and current_weights can be sent in the same transform() call
                    // Kafka Streams takes the one KeyValue<K,V> returned from transform() and sends it downstream.
    }               // it may be null

    //=========================================================================================================================

    private void updateTime() {
        long now = System.nanoTime();
        t1.set(now);
        lastActivitySeconds = Math.round(((now - t0) / 1_000_000_000.0) * 1000.0) / 1000.0;
    }

    //=========================================================================================================================
    
    private static int[] allWorkerIds(int workerId, int nWorkers, boolean includeSelf) {
        if (nWorkers <= 0) return new int[0];

        if (includeSelf) {
            int[] ids = new int[nWorkers];
            for (int i = 0; i < nWorkers; i++) ids[i] = i;  // 0 ... nWorkers - 1
            return ids;
        } else {    // here we need to explicitly exclude self. this is why nWorkers - 1
            if (nWorkers == 1) return new int[0];
            int[] ids = new int[nWorkers - 1];
            int idx = 0;
            for (int i = 0; i < nWorkers; i++) {
                if (i == workerId) continue;
                ids[idx++] = i;
            }
            return ids;
        }
    }

    //=========================================================================================================================

    private static int[] computeNeighborIds(int workerId, int nWorkers, int ringRadius, boolean includeSelf,String topology) {
        if (nWorkers <= 0) return new int[0];

        switch (topology) {
            case "square":
                int minSquare = includeSelf ? 5 : 4;
                if (nWorkers < minSquare) {
                    return allWorkerIds(workerId, nWorkers, includeSelf);
                }
                return computeSquareNeighborIds(workerId, nWorkers, includeSelf);

            case "ring":
            default:
                int maxPossible = includeSelf ? nWorkers : (nWorkers - 1);
                int requested = includeSelf ? (2 * ringRadius + 1) : (2 * ringRadius);
                if (requested > maxPossible || nWorkers < cfg.NEIGHBORHOOD_SIZE) {
                    return allWorkerIds(workerId, nWorkers, includeSelf);
                }
                return computeRingNeighborIds(workerId, nWorkers, ringRadius, includeSelf);
        }
    }

    //=========================================================================================================================

    private static int[] computeRingNeighborIds(int workerId, int nWorkers, int radius, boolean includeSelf) {
        
        if (radius <= 0) return includeSelf ? new int[]{ workerId } : new int[0];

        // Ensure we don't request more unique neighbors than exist
        radius = Math.min(radius, (nWorkers - 1) / 2);

        int size = includeSelf ? (2 * radius + 1) : (2 * radius);
        int[] ids = new int[size];
        int idx = 0;

        if (includeSelf) ids[idx++] = workerId;

        for (int d = 1; d <= radius; d++) {
            int left  = Math.floorMod(workerId - d, nWorkers);
            int right = Math.floorMod(workerId + d, nWorkers);
            ids[idx++] = left;
            ids[idx++] = right;
        }

        return ids;
    }

    //===================================================================================

    private static int[] computeSquareNeighborIds(int workerId, int nWorkers, boolean includeSelf) {
        // each node talks to its 4 von-Neumann neighbors (up/down/left/right).

        int rows = (int) Math.floor(Math.sqrt(nWorkers));   // √N_WORKERS × √N_WORKERS torus grid (2D)
        int cols = rows;
        
        // If not a perfect square, degrade gracefully to a rectangle
        if (rows * cols != nWorkers) {
            cols = (int) Math.ceil((double) nWorkers / rows); // Now rows*cols may exceed nWorkers;
        }

        // This converts a linear index (workerId) into 2D grid coordinates.
        int row = workerId / cols;
        int col = workerId % cols;

        // these should translate into worker IDs - if they dont we need to mod with nWorkers:
        int up    = Math.floorMod(row - 1, rows) * cols + col;
        int down  = Math.floorMod(row + 1, rows) * cols + col;
        int left  = row * cols + Math.floorMod(col - 1, cols);
        int right = row * cols + Math.floorMod(col + 1, cols);

        up = Math.floorMod(up, nWorkers);   // Map back into [0, nWorkers) in case rows*cols > nWorkers
        down = Math.floorMod(down, nWorkers);
        left = Math.floorMod(left, nWorkers);
        right = Math.floorMod(right, nWorkers);

        if (includeSelf) {
            return new int[]{ workerId, up, down, left, right };
        } else {
            return new int[]{ up, down, left, right };
        }
    }

    // If N_WORKERS = 16, rows = 4, cols = 4
    // maps workerId to the grid, int row = workerId / cols; int col = workerId % cols;
    // Row 0:   0   1   2   3
    // Row 1:   4   5   6   7
    // Row 2:   8   9  10  11
    // Row 3:  12  13  14  15
    // id = r * cols + c
    // stuff like Math.floorMod(row - 1, rows) is used to wrap around the grid so as to:
        // neighbors for 0 => {12, 4, 3, 1}

    // If N_WORKERS = 10, do a rectangle, rows = 3, cols = 4  
    // r=0:  0  1  2  3
    // r=1:  4  5  6  7
    // r=2:  8  9 10 11 
    // For an id of 10, 11, then it automatically becomes a 0, 1 respectively

    //=========================================================================================================================

    private List<NeighborPBest> readPBestStore() {     // for FULLY_INFORMED bestStore

        List<NeighborPBest> neighbors = new ArrayList<>();

        if (bestStore == null) {
            if (logger.isEnabled(2)) logger.log(taskInstance + 
                    ", readPBestStore: bestStore is null");
            return neighbors;
        }
        if (logger.isEnabled(1)) logger.log(taskInstance + ", pBest Weights: ");

        if(!ENABLE_NEIGHBORHOODS) {

            try (KeyValueIterator<String, ValueAndTimestamp<WeightsMessage>> it = bestStore.all()) {
                                        // this is GlobalKTable it will run for all of them
                while (it.hasNext()) {  // iterate on every Statestore (they come from different workers)
                                        // They have names: "pBest" + workerId


                    KeyValue<String, ValueAndTimestamp<WeightsMessage>> entry = it.next();
                    WeightsMessage msg = entry.value.value(); 
                    if (msg == null || msg.weights == null || msg.weights.length == 0) {
                        continue;
                    }

                    neighbors.add(new NeighborPBest(msg.weights, msg.accuracy, msg.workerId));

                    if (logger.isEnabled(1)) logger.log(msg.workerId + ")" + 
                            Dl4jParamUtils.sampleFlat(msg.weights, SAMPLING_CONSTANT) + 
                            ", with accuracy = " + msg.accuracy +  ", with loss = " + 
                            msg.loss + ", with msgIndex: " + msg.msgIndex
                    );
                }


            } catch (Exception e) {
                if (logger.isEnabled(2)) logger.log(taskInstance + 
                        ", Error iterating bestStore: " + e.getMessage());
                e.printStackTrace();
            }

            return neighbors;

        } else {    // for neighborhood, pick specific keyes

            for (int i = 0; i < neighborKeys.length; i++) {
                String key = neighborKeys[i];
                ValueAndTimestamp<WeightsMessage> value_time = bestStore.get(key);
                if (value_time == null) continue;

                WeightsMessage msg = value_time.value();
                if (msg == null || msg.weights == null || msg.weights.length == 0) continue;

                neighbors.add(new NeighborPBest(msg.weights, msg.accuracy, msg.workerId));

                if (logger.isEnabled(1)) logger.log(msg.workerId + ")" + 
                        Dl4jParamUtils.sampleFlat(msg.weights, SAMPLING_CONSTANT) + 
                        ", with accuracy = " + msg.accuracy +  ", with loss = " +   
                        msg.loss + ", with msgIndex: " + msg.msgIndex
                );
                        
            }

            return neighbors;

        }
    }


    //=========================================================================================================================

    private float[] readGBestStore() {

        if(ENABLE_NEIGHBORHOODS == true) {  // right here i am not using global gBest (bestStore.get(keyName)), but local gBest (bestStore.get(key)) 

                float minLoss = LOSS_INIT;
                WeightsMessage bestMsg = null;

                if (neighborKeys == null || neighborKeys.length == 0) {
                    return null;
                }
                if (bestStore == null) return null;

                for (String key : neighborKeys) {   // find out what the local gBest is. It may have been updated
                    ValueAndTimestamp<WeightsMessage> vat = bestStore.get(key);
                    if (vat == null) continue;

                    WeightsMessage msg = vat.value();
                    if (msg == null || msg.weights == null || msg.weights.length == 0) continue;

                    if (msg.loss < minLoss) {
                        minLoss = msg.loss;
                        bestMsg = msg;
                    }
                }

                if (bestMsg == null) return null;

                if (bestMsg.loss < ws.local_gBestLoss) {
                    ws.local_gBestLoss = bestMsg.loss;
                    ws.local_gBestAccuracy = bestMsg.accuracy;
                }
                if (bestMsg.loss < ws.stats.getLastSeenGBestLoss() - 1e-9) {
                    ws.stats.setLastSeenGBestLoss(bestMsg.loss);
                    if (logger.isEnabled(1)) logger.log(taskInstance + 
                        ", pBest Weights: " + bestMsg.workerId + ")" + 
                        Dl4jParamUtils.sampleFlat(bestMsg.weights, SAMPLING_CONSTANT) + 
                        ", with accuracy = " + bestMsg.accuracy + 
                        ", with loss = " + bestMsg.loss + ", with msgIndex: " + bestMsg.msgIndex);
                }

                return bestMsg.weights;
                

        // ===============================================================================================================================

        } else {

            if (bestStore == null) {
                if (logger.isEnabled(2)) logger.log(taskInstance + ", bestStore is null");
                return null;
            }

            ValueAndTimestamp<WeightsMessage> wrapper = bestStore.get(keyName);
            if (wrapper == null) {
                if (logger.isEnabled(1)) logger.log(taskInstance + 
                    ", gBestWeights returned null (no entry for key '" + keyName + "')");
                return null;
            }

            WeightsMessage best = wrapper.value();
            if (best == null || best.weights == null || best.weights.length == 0) {
                if (logger.isEnabled(1)) logger.log(taskInstance + 
                    ", gBestWeights is empty for key '" + keyName + "'");
                return null;
            }

            float[] gBestWeights = best.weights;
            if (gBestWeights == null || gBestWeights.length == 0) {
                if (logger.isEnabled(1)) logger.log(taskInstance + 
                    ", gBestWeights is empty for key '" + keyName + "'");
                return null;
            }

            if(best.loss < ws.local_gBestLoss) {  // update ws.local_gBestAccuracy
                ws.local_gBestLoss = best.loss;
                ws.local_gBestAccuracy = best.accuracy;
            }
            if (best.loss < ws.stats.getLastSeenGBestLoss() - 1e-9) {
                ws.stats.setLastSeenGBestLoss(best.loss);
                if (logger.isEnabled(1)) logger.log(taskInstance + 
                    ", gBest updated ! , with loss = " + best.loss + " and acc = " + best.accuracy);
            }

            return gBestWeights;
        }

    }

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
        // this has nothing to do with converging in the neighborhood
        // convergence is global this is why we use bestStore.all()

        if (bestStore == null) {
            if (logger.isEnabled(2)) logger.log(taskInstance + 
                ", [Convergence] bestStore is null");
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
                    if (logger.isEnabled(2)) logger.log(taskInstance + 
                        ", [Convergence] pBest length mismatch, skipping key = " + entry.key);
                    continue;
                }

                for (int i = 0; i < w.length; i++) sum[i] += w[i];
                n++;
            }

        } catch (Exception e) {
            if (logger.isEnabled(2)) logger.log(taskInstance + 
                ", [Convergence] error while reading pBest store: " + e.getMessage());
            e.printStackTrace();
            return null;
        }

        if (sum == null || n == 0) return null;

        float[] mean = new float[sum.length];
        for (int i = 0; i < sum.length; i++) mean[i] = (float)(sum[i] / n);
        return mean;
    }

    //=========================================================================================================================
    
    private boolean checkConvergence(boolean stricter, boolean verbose) {

            float[] center = null;

            if (FULLY_INFORMED == false) {

                center = readGBestStore();
                if (center == null) {
                    if (logger.isEnabled(2)) logger.log(taskInstance + 
                        ", [Convergence] gBest not available -> cannot evaluate convergence.");
                    return false;
                }

            } else {

                center = computeMeanPBestFromStore();
                if (center == null) {
                    if (logger.isEnabled(2)) logger.log(taskInstance + 
                        ", [Convergence] pBest mean not available -> cannot evaluate convergence.");
                    return false;
                }
            }

            double dist = normL2PerDim(ws.flatModel, center);       // dist ≈ 0.05 → each weight differs by ~0.05 on average
                                                                    // this is a normalized / scaled metric 

            // ==========================================================================================================
            double radius;

            if(stricter) {
                radius = CONVERGENCE_STRICTNESS_FACTOR * CONVERGENCE_ALPHA * Dl4jParamUtils.rms(center);    // rms measures magnitude 
            } else {
                radius = CONVERGENCE_ALPHA * Dl4jParamUtils.rms(center); 
                    // rms(center) is the typical magnitude of the weights in the center model.
                    // The particle is converged if, on average, each weight differs from the center by 
                    // less than CONVERGENCE_ALPHA * 100% (i.e. 10%) of a typical weight’s magnitude.

                    // So condition:
                    // converge if rms(x - center) <= α * rms(center)
                    // That is basically a relative closeness test:
                    // converge if rms(x - center) / rms(center) <= α

            }

            // ==========================================================================================================
            // double radius = CONVERGENCE_ALPHA * Dl4jParamUtils.rms(center);
            // ==========================================================================================================

            boolean converged = dist <= radius;

            if(verbose) {
                if (logger.isEnabled(2)) logger.log(taskInstance + " dist = " + String.format("%.4f", dist)
                        + " radius = " + Dl4jParamUtils.round((float) radius, 4)
                        + ", with velocity (magnitude): " + Dl4jParamUtils.rmsScaled(ws.velocity, 100) 
                        + " => " + (converged ? "CONVERGED" : "NOT_CONVERGED"));

                System.out.println("[Worker " + workerId + "] Dist = " + String.format("%.4f", dist)
                        + ", Radius = " + Dl4jParamUtils.round((float) radius, 4)
                        + ", with velocity (magnitude): " + Dl4jParamUtils.rmsScaled(ws.velocity, 100) 
                        + " => " + (converged ? "CONVERGED" : "NOT_CONVERGED"));
            }

            return converged;
    }
    //=========================================================================================================================

    private void flushPendingPBest() {

        if (pendingPBestMsg == null) return;

        long nowMs = System.currentTimeMillis();
        if (nowMs - openedDebounceWindow  < cfg.PBEST_DEBOUNCE_MS) return; // if there is no significant difference between
            // when last sent Best / when did I last forward a pBest downstream
            // don’t forward more than once per debounce interval

        // Latest-wins
        WeightsMessage msg = pendingPBestMsg;

        pendingPBestMsg = null;
        pBestUpdatePending = false;
        lastPBestForwardMs = nowMs;

        // This is the actual emission downstream from the Transformer
        ws.pBestForwardedCount++;
        ws.incrementTotalMessagesSent("pBest");

        context.forward(keyName, msg);

        if (logger.isEnabled(1)) {
            logger.log(taskInstance + " Forwarded pBest key = " + keyName +
                    " msgIndex = " + msg.msgIndex + " loss = " + msg.loss + " acc = " + msg.accuracy);
        }
    }

    //=========================================================================================================================

    private void updateThreshold(int t) {     // threshold(t)=threshold_min + (threshold_max - threshold_min) * exp(-t / tau)

        int tc = Math.min(t, MAX_UPDATES);

        double expTerm = Math.exp(-(double) tc / (double) TAU);     // -t/tau
        double threshold = (double) LOSS_THRESHOLD_MIN + ((double) LOSS_THRESHOLD_MAX - (double) LOSS_THRESHOLD_MIN) * expTerm;

        if (threshold < LOSS_THRESHOLD_MIN) threshold = LOSS_THRESHOLD_MIN;
        if (threshold > LOSS_THRESHOLD_MAX) threshold = LOSS_THRESHOLD_MAX;

        this.loss_threshold = (float) threshold;
    }

    //=========================================================================================================================

    private void updateMonitoringThreshold(int iter) {
        // iter: ws.countForwardPasses (global per worker)
        int T = Math.max(1, MAX_UPDATES);
        double p = Math.min(1.0, (double) iter / (double) T);

        double val = MONITORING_THRESHOLD_MAX + (MONITORING_THRESHOLD_MIN - MONITORING_THRESHOLD_MAX) * p;

        monitoring_threshold = (int) Math.round(val);
        logger.log("monitoring_threshold: " + monitoring_threshold + ", MONITORING_THRESHOLD_MAX: " +
            MONITORING_THRESHOLD_MAX + ", MONITORING_THRESHOLD_MIN: " + MONITORING_THRESHOLD_MIN
        );
    }

    //=========================================================================================================================

    @Override
    public void close() {

        ws.countPartitionsFinished += 1;
        if (!buffer.isEmpty()) {
            buffer.clear();
        }
        double totalElapsedTimeSec = (System.nanoTime() - this.t0) / 1_000_000_000.0;  // in Seconds

        if(lastOffset == 0) {   // if inactive Partition, means Worker terminated before starting to read that partition 
                                // (worker reads the partitions with a limited degree of parallelism, not 40 at once)
                                // limited degree of parallelism => depends on num.stream.threads. Some tasks never get scheduled
                                // Each stream thread processes one task at a time => limited concurrent partition processing overall
            ws.inactivePartitions++;
            if (logger.isEnabled(2)) logger.log(taskInstance + ", Empty partition: " 
                + seenPartitions + ", with lastOffset: " + lastOffset + 
                ", inactivePartitions so far: " + ws.inactivePartitions);

        } else {    // if active partition
        
            if (logger.isEnabled(2)) logger.log(taskInstance + ", Seen partitions: " 
                + seenPartitions + ", with lastOffset: " + lastOffset);

            double avgMs = (sumElapsedNs / 1_000_000.0) / per_task_count;    // this is the overall time of processing a batch
            double avgMsUpdateX = (sumElapsedNsUpdateX / 1_000_000.0) / per_task_count; 
            double avgMsPredict = (sumElapsedNsPredict / 1_000_000.0) / per_task_count; 
            double avgForwardPassMs = forwardPassNs / countForwardPass;     // this is just the forward pass part of it (1 batch => 1 forward pass)
                            // what we are observing is that forward pass takes the most amount of time inside the entire batch processing
            
            if (logger.isEnabled(2)) logger.log(taskInstance + 
                    ", average elapsed time Measurements: over " + per_task_count + " batches: " + "\n" +
                    "=> per batch: " + String.format("%.3f ms", avgMs) + "\n" + 
                    "=> per updateX: " + String.format("%.3f ms", avgMsUpdateX) + "\n" + 
                    "=> per Prediction: " + String.format("%.3f ms", avgMsPredict) + "\n" + 
                    "   => per forwardPassMs: " + avgForwardPassMs + "\n" + 
                    "Rate of Updates / Batches per sec: " + String.format("%.5f sec", per_task_count / totalElapsedTimeSec)   // this is count_of_updates per seconds
                        // Also equivalent with batches per second
            );

            ws.validAvgMs = avgMs;
        } 

        // Report on convergence: ==============================================================

        if(ws.printedReport == false) {

            
            if (logger.isEnabled(2)) logger.log("Opening Report ============================================================");
            System.out.println(PredictorComparisonRegistry.summary());
            
            checkConvergence(false, true);

            if(countForwardPass != 0) {
                if (logger.isEnabled(2)) logger.log("Average bufferSize: " + Dl4jParamUtils.round(bufferSizeAcc / countForwardPass, 2));
            }            

            if (logger.isEnabled(2)) logger.log("neighborKeys: " + Arrays.toString(neighborKeys));
    
            if (logger.isEnabled(2)) logger.log("FIltering Statistics:");
            if (logger.isEnabled(2)) logger.log("TOTAL_MESSAGES_SENT: "+ ws.TOTAL_MESSAGES_SENT +
                 ", TOTAL_BYTES_SENT: " + ws.TOTAL_BYTES_SENT);

            if (logger.isEnabled(2)) logger.log("improved_pBest_count: "+ ws.improved_pBest_count + ", significant_pBest_count: " + ws.significant_pBest_count);
            if (logger.isEnabled(2)) logger.log("pBestCandidateCount: "+ ws.pBestCandidateCount + ", pBestForwardedCount: " + ws.pBestForwardedCount);

            if (logger.isEnabled(2)) logger.log("incrementBatchesRead: " + ws.incrementBatchesRead + ", incrementSamplesRead: " + ws.incrementSamplesRead );
            if (logger.isEnabled(2)) logger.log("printBatchesReadSummary: " + WorkerStatic.printBatchesReadSummary());

            if (logger.isEnabled(2)) logger.log("Closing Report ============================================================");

            ws.printedReport = true;
        }

        logger.log("countPartitionsFinished: " + ws.countPartitionsFinished + 
            ", numberOfTasks: " + ws.numberOfTasks + "\n");

        if(ws.countPartitionsFinished == ws.numberOfTasks - 5) {    // these 5 are not normal tasks
                        // there are always 5 extra control threads
            if(logger.isEnabled(2)) logger.log("Final inActivePartitions: " + ws.inactivePartitions);
            if(logger.isEnabled(2)) logger.log("Number of forward passes: " + 
                ws.countForwardPasses + ", globally: " + countForwardPassesStatic);

            System.out.println("[Worker " + workerId + "] Average Elapsed Time per Batch: " +
                    String.format("%.3f ms", ws.validAvgMs) + ", InActivePartitions " + ws.inactivePartitions);
        }

        logger.flush();
    }
}


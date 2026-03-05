package transformers;

import org.apache.kafka.streams.kstream.Transformer;
import org.apache.kafka.streams.KeyValue;
import org.apache.kafka.streams.processor.ProcessorContext;
import org.apache.kafka.streams.state.KeyValueStore;

import dl4j_models.Dl4jParamUtils;
import message.weights_message.WeightsMessage;
import utils.CustomLogger;
import utils.Config;
import java.util.concurrent.atomic.AtomicInteger;

public class GBestTransformer implements Transformer<String, WeightsMessage, KeyValue<String, WeightsMessage>> {

    private final CustomLogger logger;

    private static Config cfg = Config.getInstance();
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;
    private final float SIGNIFICANT_LOSS_DIFF = cfg.SIGNIFICANT_LOSS_DIFF;

    private ProcessorContext context;
    private KeyValueStore<String, Float> gBestLossStore;

    private float gBestLoss = 100000f;

    private static final AtomicInteger INSTANCE_SEQ = new AtomicInteger(0);
    private final int instanceNo = INSTANCE_SEQ.incrementAndGet();
    private final String taskInstance = instanceNo + "@" + Integer.toHexString(System.identityHashCode(this));

    private String taskTag = "task=UNKNOWN";

    private static final float EPS = 1e-9f;

    private float lastSentGBestLoss = 100000f;

    private long t0;
    private double lastActivitySeconds = 0.0;

    private int count = 0;

    // =====================================================================================================================

    public GBestTransformer(CustomLogger logger, long t0) {
        this.logger = logger;
        this.t0 = t0;
    }

    // =====================================================================================================================

    @Override
    @SuppressWarnings("unchecked")
    public void init(ProcessorContext context) {
        this.context = context;
        this.gBestLossStore = (KeyValueStore<String, Float>) context.getStateStore("gBestEmitStore");
    }

    // =====================================================================================================================

    @Override
    public KeyValue<String, WeightsMessage> transform(String key, WeightsMessage msg) {
        
        if(count == 0) {
            logger.log("[gBest Logger] Starting Delay: " + (System.nanoTime() - this.t0) / 1_000_000_000.0);
        }
        count++;
        updateTime();

        if (msg == null) return null;

        float newLoss = msg.loss;

        if (newLoss < (gBestLoss - EPS)) {

            gBestLoss = newLoss;
            gBestLossStore.put("gBestLoss", gBestLoss);

            WeightsMessage gBestMsg = new WeightsMessage(msg.workerId, msg.msgIndex, msg.accuracy, msg.loss, msg.weights);

            if (logger.isEnabled(1)) logger.log(lastActivitySeconds + 
                ", [gBest updated] workerId = " + msg.workerId + ", accuracy = " + msg.accuracy +
                ", loss = " + msg.loss + ", with weights: " + Dl4jParamUtils.sampleFlat(msg.weights, SAMPLING_CONSTANT));

            boolean significant_improvement = Math.abs((lastSentGBestLoss - newLoss)) / (Math.abs(lastSentGBestLoss) + EPS) > 0.3 * SIGNIFICANT_LOSS_DIFF;

            if (!significant_improvement) {
                return null;
            }
            
            return new KeyValue<>("gBest", gBestMsg);

        
        }

        return null;
    }

    // =====================================================================================================================

    @Override
    public void close() { 

    }

    // ==============================================================================================

    private void updateTime() {
        lastActivitySeconds = Math.round(((System.nanoTime() - t0) / 1_000_000_000.0) * 1000.0) / 1000.0;
    }
}

package transformers;

import org.apache.kafka.streams.processor.api.Processor;
import org.apache.kafka.streams.processor.api.ProcessorContext;
import org.apache.kafka.streams.processor.api.Record;
import org.apache.kafka.streams.state.KeyValueStore;

import message.weights_message.WeightsMessage;
import utils.CustomLogger;
import utils.Dl4jParamUtils;

import utils.*;

public class GBestProcessor implements Processor<String, WeightsMessage, String, WeightsMessage> {

    private final CustomLogger logger;
    private static Config cfg = Config.getInstance();
    private static final int SAMPLING_CONSTANT = cfg.SAMPLING_CONSTANT;

    private ProcessorContext<String, WeightsMessage> context;
    private KeyValueStore<String, Float> gBestLossStore;

    private float gBestLoss = 100000f;

    public GBestProcessor(CustomLogger logger) {
        this.logger = logger;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void init(ProcessorContext<String, WeightsMessage> context) {
        this.context = context;
        this.gBestLossStore = (KeyValueStore<String, Float>) context.getStateStore("gBestEmitStore");
    }

    @Override
    public void process(Record<String, WeightsMessage> record) {

        WeightsMessage msg = record.value();
        if (msg == null) return;

        float newLoss = msg.loss;

        final float EPS = 1e-9f;

        if (newLoss < (gBestLoss - EPS)) {
            gBestLoss = newLoss;
            gBestLossStore.put("gBestLoss", gBestLoss);

            WeightsMessage gBestMsg = new WeightsMessage(msg.workerId, msg.msgIndex, msg.accuracy, msg.loss, msg.weights);

            context.forward(new Record<>("gBest", gBestMsg, record.timestamp()));

            logger.log("[gBest updated] workerId = " + msg.workerId + ", accuracy = " + msg.accuracy + ", loss = " + msg.loss
                        + ", with weights: " + Dl4jParamUtils.sampleFlat(msg.weights, SAMPLING_CONSTANT));
        }

        // else: drop, dont write to GLOBAL_WEIGHTS_TOPIC
    }

    @Override
    public void close() { }
}

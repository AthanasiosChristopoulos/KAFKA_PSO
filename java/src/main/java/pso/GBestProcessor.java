package pso;

import org.apache.kafka.streams.processor.api.Processor;
import org.apache.kafka.streams.processor.api.ProcessorContext;
import org.apache.kafka.streams.processor.api.Record;
import org.apache.kafka.streams.state.KeyValueStore;

import message.weights_message.WeightsMessage;
import utils.CustomLogger;

public class GBestProcessor implements Processor<String, WeightsMessage, String, WeightsMessage> {

    private final CustomLogger logger;

    private ProcessorContext<String, WeightsMessage> context;
    private KeyValueStore<String, Float> gBestLossStore;

    private float gBestLoss = Float.POSITIVE_INFINITY;

    public GBestProcessor(CustomLogger logger) {
        this.logger = logger;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void init(ProcessorContext<String, WeightsMessage> context) {
        this.context = context;
        this.gBestLossStore = (KeyValueStore<String, Float>) context.getStateStore("gBestEmitStore");

        Float persisted = gBestLossStore.get("gBestLoss");
        if (persisted != null) {
            gBestLoss = persisted;
        }
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

            WeightsMessage gBestMsg = new WeightsMessage(msg.idWorker, msg.msgIndex, msg.accuracy, msg.loss, msg.weights);

            context.forward(new Record<>("gBest", gBestMsg, record.timestamp()));

            logger.log("[gBest updated+sent] workerId=" + msg.idWorker
                + ", msgIndex=" + msg.msgIndex
                + ", acc=" + msg.accuracy
                + ", loss=" + msg.loss);
        }

        // else: drop, dont write to GLOBAL_WEIGHTS_TOPIC
    }

    @Override
    public void close() { }
}

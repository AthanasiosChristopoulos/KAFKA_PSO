package pso;

import org.apache.kafka.streams.kstream.Transformer;
import org.apache.kafka.streams.KeyValue;
import org.apache.kafka.streams.processor.ProcessorContext;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class BatchingTransformer implements Transformer<String, String, KeyValue<String, String>> {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final int workerId;
    private final int batchSize;
    private final int nBatches;
    private final WorkerSharedState shared;

    private ProcessorContext context;

    private final List<String> buffer = new ArrayList<>();

    private final MultiLayerNetwork model;
    private final WorkerStats stats;
    private final BatchPrediction predictor;
    private final PsoUpdater psoUpdater;

    private double[] pBestWeights;

    private int batchesRead = 0;

    private int count = 0;

    public BatchingTransformer(int workerId, WorkerSharedState shared, int batchSize, int nBatches) {

        this.workerId = workerId;
        this.shared = shared;
        this.batchSize = batchSize;
        this.nBatches = nBatches;

        this.model = Dl4jModelFactory.createIrisModel();
        this.stats = new WorkerStats();
        this.predictor = new BatchPrediction(model, stats);
        this.psoUpdater = new PsoUpdater(model, workerId);

        this.pBestWeights = Dl4jParamUtils.modelToFlatList(model);
        shared.gBestWeights = Dl4jParamUtils.modelToFlatList(model);
    }

    @Override
    public void init(ProcessorContext context) {
        this.context = context;
    }

    @Override
    public KeyValue<String, String> transform(String key, String value) {

        if(count == 0) {
            shared.log("Starting Meta Data: " + context.topic() + ", Partition: " + context.partition()+ ", Offset: " + context.offset());
        }
        count = count + 1;

        if (value == null) {
            return null;
        }

        buffer.add(value);

        if (buffer.size() < batchSize) {
            return null;
        }

        predictor.callPredictionsBatch(buffer);
        buffer.clear();
        batchesRead++;

        double accuracy = stats.getAccuracy();
        if (accuracy == -1) {
            return null;
        }

        double[] weights = Dl4jParamUtils.modelToFlatList(model);
        List<Double> weightList = new ArrayList<>(weights.length);
        for (double v : weights) {
            weightList.add(v);
        }

        if (accuracy > stats.getBestAccuracy()) {

            stats.setBestAccuracy(accuracy);
            this.pBestWeights = weights;

            shared.log("Improved accuracy to: " + stats.getBestAccuracy() + ", n_predictions: " + stats.getNumPredictions() +
                       ", n_correct: " + stats.getNumCorrect());

            var payload = new HashMap<String, Object>();
            payload.put("pBestMsgIndex", java.util.UUID.randomUUID().toString());
            payload.put("id_worker", workerId);
            payload.put("pBest", weightList);
            payload.put("accuracy", accuracy);

            try {
                String json = MAPPER.writeValueAsString(payload);
                shared.log("SENDING pBest JSON: " + json);

                // emit pBest downstream (LOCAL_WEIGHTS_TOPIC)
                return new KeyValue<>("1", json);

            } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
                e.printStackTrace();
                return null;
            }
        }

        if (batchesRead >= nBatches) {
            if(shared.gBestWeightsChanged == 0) {
                System.out.println("No gBestWeights received");
            }
            
            try {
                var payload = new HashMap<String, Object>();
                payload.put("id_worker", this.workerId);
                payload.put("weightsMsgIndex", java.util.UUID.randomUUID().toString());
                payload.put("weights", weightList);

                String json = MAPPER.writeValueAsString(payload);

                batchesRead = 0;

                return new KeyValue<>("1", json);

            } catch (Exception e) {
                e.printStackTrace();
                return null;
            }
        }

        double[] velocity = psoUpdater.updateX(model, this.pBestWeights, shared.gBestWeights);
        stats.reset();

        shared.log("Updated Model to: " + Dl4jParamUtils.sampleFlat(Dl4jParamUtils.modelToFlatList(model)) +
                    ", with velocity: " + Dl4jParamUtils.sampleFlat(velocity));
    
        return null;
    }

    @Override
    public void close() {
        if (!buffer.isEmpty()) {
            predictor.callPredictionsBatch(buffer);
            buffer.clear();
        }
    }
}

package utils;

import org.apache.kafka.common.serialization.Deserializer;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Map;

public class WeightsMessageDeserializer implements Deserializer<WeightsMessage> {

    @Override
    public WeightsMessage deserialize(String topic, byte[] bytes) {
        
        if (bytes == null) {
            return null;
        }

        ByteBuffer buffer = ByteBuffer.wrap(bytes);

        int idWorker = buffer.getInt();

        int msgLen = buffer.getInt();
        byte[] msgBytes = new byte[msgLen];
        buffer.get(msgBytes);
        String msgIndex = new String(msgBytes, StandardCharsets.UTF_8);

        float accuracy = buffer.getFloat();
        float loss = buffer.getFloat();

        int nWeights = buffer.getInt();
        float[] weights = new float[nWeights];
        for (int i = 0; i < nWeights; i++) {
            weights[i] = buffer.getFloat();
        }

        return new WeightsMessage(idWorker, msgIndex, accuracy, loss, weights);
    }

    @Override
    public void configure(Map<String, ?> configs, boolean isKey) {
        // no-op
    }

    @Override
    public void close() {
        // no-op
    }
}

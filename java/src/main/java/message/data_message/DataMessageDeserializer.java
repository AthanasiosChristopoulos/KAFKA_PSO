package utils;

import org.apache.kafka.common.serialization.Deserializer;

import java.nio.ByteBuffer;
import java.util.Map;

public class DataMessageDeserializer implements Deserializer<DataMessage> {

    @Override
    public DataMessage deserialize(String topic, byte[] bytes) {
        if (bytes == null) return null;

        try {
            ByteBuffer buffer = ByteBuffer.wrap(bytes);

            int sampleIndex = buffer.getInt();
            int label = buffer.getInt();

            int nFeatures = buffer.getInt();
            if (nFeatures < 0) {
                throw new IllegalArgumentException("Negative nFeatures: " + nFeatures);
            }

            // basic sanity check to avoid BufferUnderflow on corrupt messages
            int remaining = buffer.remaining();
            int needed = nFeatures * 4;
            if (remaining < needed) {
                throw new IllegalArgumentException(
                    "Not enough bytes for features: need " + needed + " but have " + remaining
                );
            }

            float[] features = new float[nFeatures];
            for (int i = 0; i < nFeatures; i++) {
                features[i] = buffer.getFloat();
            }

            return new DataMessage(sampleIndex, features, label);

        } catch (Exception e) {
            // If you prefer "fail fast", rethrow. If you prefer "drop bad records", return null.
            return null;
        }
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

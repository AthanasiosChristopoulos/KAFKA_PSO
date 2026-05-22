package message.data_message;

import org.apache.kafka.common.serialization.Serializer;

import java.nio.ByteBuffer;
import java.util.Map;

public class DataMessageSerializer implements Serializer<DataMessage> {

    @Override
    public byte[] serialize(String topic, DataMessage data) {
        if (data == null) return null;

        int sampleIndex = data.sampleIndex;
        int label = data.label;

        float[] features = (data.features == null) ? new float[0] : data.features;
        int nFeatures = features.length;

        int size = 4 + 4 + 4 + nFeatures * 4;

        ByteBuffer buffer = ByteBuffer.allocate(size);
        buffer.putInt(sampleIndex);
        buffer.putInt(label);
        buffer.putInt(nFeatures);

        for (int i = 0; i < nFeatures; i++) {
            buffer.putFloat(features[i]);
        }

        return buffer.array();
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

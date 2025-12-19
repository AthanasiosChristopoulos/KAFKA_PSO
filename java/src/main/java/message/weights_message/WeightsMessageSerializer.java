package message;

import org.apache.kafka.common.serialization.Serializer;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Map;

public class WeightsMessageSerializer implements Serializer<WeightsMessage> {

    @Override
    public byte[] serialize(String topic, WeightsMessage data) { // Convert WeightsMessage into bytes

        if (data == null) {
            return null;
        }

        int idWorker = data.idWorker;
    
        String msgIndex = data.msgIndex;

        byte[] msgBytes = msgIndex.getBytes(StandardCharsets.UTF_8);    // calculate strings size as bytes
        int msgLen = msgBytes.length;   

        float accuracy = data.accuracy;
        float loss = data.loss;

        float[] w = data.weights;
        int nWeights = w.length;

        // total size in bytes (float == 4 Bytes):
        // int idWorker + int msgLen + msgLen bytes + float accuracy
        // + float loss + int nWeights + nWeights * float

        int size = 4 + 4 + msgLen + 4 + 4 + 4 + nWeights * 4;

        ByteBuffer buffer = ByteBuffer.allocate(size);

        buffer.putInt(idWorker);
        buffer.putInt(msgLen);
        buffer.put(msgBytes);
        buffer.putFloat(accuracy);
        buffer.putFloat(loss);
        buffer.putInt(nWeights);

        for (int i = 0; i < nWeights; i++) {
            buffer.putFloat(w[i]);
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

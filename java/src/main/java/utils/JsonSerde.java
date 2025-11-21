package utils;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.kafka.common.serialization.Deserializer;
import org.apache.kafka.common.serialization.Serde;
import org.apache.kafka.common.serialization.Serializer;

public class JsonSerde<T> implements Serde<T> {

    private final ObjectMapper mapper = new ObjectMapper();
    private final Class<T> type;

    public JsonSerde(Class<T> type) {
        this.type = type;
    }

    @Override
    public Serializer<T> serializer() {
        return (topic, data) -> {
            try {
                if (data == null) return null;
                return mapper.writeValueAsBytes(data);
            } catch (Exception e) {
                throw new RuntimeException("JSON Serialization failed", e);
            }
        };
    }

    @Override
    public Deserializer<T> deserializer() {
        return (topic, bytes) -> {
            try {
                if (bytes == null) return null;
                return mapper.readValue(bytes, type);
            } catch (Exception e) {
                throw new RuntimeException("JSON Deserialization failed", e);
            }
        };
    }
}

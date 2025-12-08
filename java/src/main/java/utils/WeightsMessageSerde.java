package utils;

import org.apache.kafka.common.serialization.Serdes;

public class WeightsMessageSerde extends Serdes.WrapperSerde<WeightsMessage> {
    
    public WeightsMessageSerde() {
        super(new WeightsMessageSerializer(), new WeightsMessageDeserializer());
    }

    // this is a Serde (WeightsMessageSerde). It contains both the serializer and the deserializer
}

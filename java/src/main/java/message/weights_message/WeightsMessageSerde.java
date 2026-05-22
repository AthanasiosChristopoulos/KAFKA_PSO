package message.weights_message;

import org.apache.kafka.common.serialization.Serdes;

public class WeightsMessageSerde extends Serdes.WrapperSerde<WeightsMessage> {
    
    public WeightsMessageSerde() {
        super(new WeightsMessageSerializer(), new WeightsMessageDeserializer());
    }
}

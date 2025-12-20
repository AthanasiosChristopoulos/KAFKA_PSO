package message;

import org.apache.kafka.common.serialization.Serdes;

public class DataMessageSerde extends Serdes.WrapperSerde<DataMessage> {
    
    public DataMessageSerde() {
        super(new DataMessageSerializer(), new DataMessageDeserializer());
    }
}

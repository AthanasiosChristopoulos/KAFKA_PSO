
package message.data_message;

import java.util.List;

public class DataMessage {
    public int sampleIndex;
    public float[] features;
    public int label;

    public DataMessage(int sampleIndex, float[] features, int label) {
        this.sampleIndex = sampleIndex;
        this.features = features;
        this.label = label;
    }

}

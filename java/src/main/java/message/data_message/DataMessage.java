
package message;

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

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("sampleIndex=").append(sampleIndex);
        sb.append(", label=").append(label);

        sb.append(", featuresSample=[");

        int n = Math.min(5, features.length);
        for (int i = 0; i < n; i++) {
            sb.append(String.format("%.5f", features[i]));
            if (i < n - 1) sb.append(", ");
        }
        
        sb.append(", ...]}");
        return sb.toString();
    }

}

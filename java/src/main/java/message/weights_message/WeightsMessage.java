
package message.weights_message;

public class WeightsMessage {
    public int workerId;
    public String msgIndex;
    public float accuracy;
    public float loss;
    public float[] weights;
    public long timestamp;

    public WeightsMessage(int workerId, String msgIndex, float accuracy, float loss, float[] weights, long timestamp) {
        this.workerId = workerId;
        this.msgIndex = msgIndex;
        this.accuracy = accuracy;
        this.loss = loss;
        this.weights = weights;
        this.timestamp = timestamp;
    }

}

// WeightsMessage size estimation (approximately for dimensionality = 20000)

// int workerId = 4 bytes
// String msgIndex ≈ 40 bytes (object + char[] overhead, typical small string)
// float accuracy = 4 bytes
// float loss = 4 bytes
// float[] weights (length = 20000)
//   each float = 4 bytes
//   20000 * 4 = 80000 bytes
//   array overhead ≈ 16 bytes
//   total weights ≈ 80016 bytes

// Total ≈ 4 + 40 + 4 + 4 + 80016 = 80068 bytes

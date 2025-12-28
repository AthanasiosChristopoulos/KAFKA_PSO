
package message.weights_message;

public class WeightsMessage {
    public int workerId;
    public String msgIndex;
    public float accuracy;
    public float loss;
    public float[] weights;

    public WeightsMessage(int workerId, String msgIndex, float accuracy, float loss, float[] weights) {
        this.workerId = workerId;
        this.msgIndex = msgIndex;
        this.accuracy = accuracy;
        this.loss = loss;
        this.weights = weights;
    }

}


package message.weights_message;

public class WeightsMessage {
    public int idWorker;
    public String msgIndex;
    public float accuracy;
    public float loss;
    public float[] weights;

    public WeightsMessage(int idWorker, String msgIndex, float accuracy, float loss, float[] weights) {
        this.idWorker = idWorker;
        this.msgIndex = msgIndex;
        this.accuracy = accuracy;
        this.loss = loss;
        this.weights = weights;
    }

}

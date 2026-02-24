package experimentation;

public final class WorkerMetrics {
    private final int workerId;
    private final double elapsedSec;
    private final float bestAccuracy;
    private final float bestLoss;

    public WorkerMetrics(int workerId, double elapsedSec, float bestAccuracy, float bestLoss) {
        this.workerId = workerId;
        this.elapsedSec = elapsedSec;
        this.bestAccuracy = bestAccuracy;
        this.bestLoss = bestLoss;
    }

    public int getWorkerId() { return workerId; }
    public double getElapsedSec() { return elapsedSec; }
    public float getBestAccuracy() { return bestAccuracy; }
    public float getBestLoss() { return bestLoss; }

    @Override
    public String toString() {
        return "WorkerMetrics{" +
                "workerId=" + workerId +
                ", elapsedSec=" + elapsedSec +
                ", bestAccuracy=" + bestAccuracy +
                ", bestLoss=" + bestLoss +
                '}';
    }
}
package experimentation;

public final class WorkerMetrics {
    private final int workerId;
    private final double elapsedSec;
    private final float bestAccuracy;
    private final float bestLoss;
    public final long TOTAL_MESSAGES_SENT;
    public final long TOTAL_BYTES_SENT;

    public WorkerMetrics(int workerId, double elapsedSec, float bestAccuracy, float bestLoss, long TOTAL_MESSAGES_SENT, long TOTAL_BYTES_SENT) {
        this.workerId = workerId;
        this.elapsedSec = elapsedSec;
        this.bestAccuracy = bestAccuracy;
        this.bestLoss = bestLoss;
        this.TOTAL_MESSAGES_SENT = TOTAL_MESSAGES_SENT;
        this.TOTAL_BYTES_SENT = TOTAL_BYTES_SENT;
    }

    public int getWorkerId() { return workerId; }
    public double getElapsedSec() { return elapsedSec; }
    public float getBestAccuracy() { return bestAccuracy; }
    public float getBestLoss() { return bestLoss; }
    public long getTOTAL_MESSAGES_SENT() { return TOTAL_MESSAGES_SENT; }
	public long getTOTAL_BYTES_SENT() { return TOTAL_BYTES_SENT; }

	@Override public String toString() {
        return "WorkerMetrics{" +
                "workerId=" + workerId +
                ", elapsedSec=" + elapsedSec +
                ", bestAccuracy=" + bestAccuracy +
                ", bestLoss=" + bestLoss +
                '}';
    }
}
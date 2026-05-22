package state;

public class Stats {
    private float bestAccuracy = 0f;
    private float bestLoss = 100000f;
    private float lastSentPBestLoss = 100000f;
    private float lastSeenGBestLoss = 100000f;

    public float getBestAccuracy() { return Math.round(bestAccuracy * 1000f) / 1000f; }
    public void setPBestAccuracy(float a) { bestAccuracy = a; }

    public float getPBestLoss() { return Math.round(bestLoss * 1000f) / 1000f; }
    public void setPBestLoss(float l) { bestLoss = l; }

    public float getLastSentPBestLoss() { return Math.round(lastSentPBestLoss * 1000f) / 1000f; }
    public void setLastSentPBestLoss(float l) { lastSentPBestLoss = l; }

    public float getLastSeenGBestLoss() {
		return lastSeenGBestLoss;
	}

	public void setLastSeenGBestLoss(float lastSeenGBestLoss) {
		this.lastSeenGBestLoss = lastSeenGBestLoss;
	}
}

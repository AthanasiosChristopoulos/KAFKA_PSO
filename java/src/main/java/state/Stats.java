package state;

public class Stats {

    private long nSamples = 0;
    private long nCorrect = 0;
    private float loss = 0;

    private float bestAccuracy = 0;
    private float bestLoss = 100000f;
    private float lastSentPBestLoss = 100000f;
    private float lastSeenGBestLoss = 100000f;

    public void addBatch(int nSamples, int nCorrect, float loss) {
        if(nSamples == 0) {
            System.out.println("0 Samples");
        }

        this.nSamples += nSamples;
        this.nCorrect += nCorrect;
        this.loss += loss;
    }

	public long getNumPredictions() {
        return nSamples;
    }

    public long getNumCorrect() {
        return nCorrect;
    }

    // Accuracy ==========================================================================

    public float getPAccuracy() {
        
        float acc = (float) nCorrect / nSamples;

        if (nSamples == 0) {
            System.out.println("0f accuracy, because nSamples == 0, nCorrect: " + nCorrect);
            return 0f;
        } 
        if (nCorrect == 0) {
            System.out.println("0f accuracy, because nCorrect == 0 , nSamples: " + nSamples);
            return 0f;
        } 
        if (acc == 0) {
            System.out.println("0f accuracy, because acc == 0");
            return 0f;
        } 

        return Math.round(acc * 1000f) / 1000f; // round at 3 decimal positions 
    }

    public float getBestAccuracy() {
        return Math.round(bestAccuracy * 1000f) / 1000f;
    }
    
    public void setPBestAccuracy(float newBestAccuracy) {
        this.bestAccuracy = newBestAccuracy;
    }

    // Loss ==========================================================================

    public float getLoss() {
        return Math.round(loss * 1000f) / 1000f; 
    }
    
    public float getPBestLoss() {
        return Math.round(bestLoss * 1000f) / 1000f; 
    }
    
    public void setPBestLoss(float bestLoss) {
        this.bestLoss = bestLoss;
    }

    public float getLastSentPBestLoss() {
        return Math.round(lastSentPBestLoss * 1000f) / 1000f; 
    }
    
    public void setLastSentPBestLoss(float lastSentPBestLoss) {
        this.lastSentPBestLoss = lastSentPBestLoss;
    }

    public float getLastSeenGBestLoss() {
		return lastSeenGBestLoss;
	}

	public void setLastSeenGBestLoss(float lastSeenGBestLoss) {
		this.lastSeenGBestLoss = lastSeenGBestLoss;
	}
    // ================================================================================

    public void reset() {
        nSamples = 0;
        nCorrect = 0;
        loss = 0;
    }
    
}

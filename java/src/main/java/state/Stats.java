package state;

public class Stats {

    private long nSamples = 0;
    private long nCorrect = 0;
    private float loss = 0;

    private float bestAccuracy = 0;
    private float bestLoss = 1000000;

    public void addBatch(int nSamples, int nCorrect, float loss) {
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

    public float getAccuracy() {
        if (nSamples == 0) {
            System.out.println("0f, because no Predictions");
            return 0f;
        } 

        float acc = (float) nCorrect / nSamples;
        return Math.round(acc * 1000f) / 1000f; // round at 3 decimal positions 
    }

    public float getBestAccuracy() {
        return Math.round(bestAccuracy * 1000f) / 1000f;
    }
    
    public void setBestAccuracy(float newBestAccuracy) {
        this.bestAccuracy = newBestAccuracy;
    }

    // Loss ==========================================================================

    public float getLoss() {
        return Math.round(loss * 1000f) / 1000f; 
    }
    
    public float getBestLoss() {
        return Math.round(bestLoss * 1000f) / 1000f; 
    }
    
    public void setBestLoss(float bestLoss) {
        this.bestLoss = bestLoss;
    }

    // ================================================================================

    public void reset() {
        nSamples = 0;
        nCorrect = 0;
        loss = 0;
    }
    
}

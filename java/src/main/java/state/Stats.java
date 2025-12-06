package state;

public class Stats {

    private long nSamples = 0;
    private long nCorrect = 0;
    private double loss = 0;

    private double bestAccuracy = 0;
    private double bestLoss = 1000000;

    public void addBatch(int nSamples, int nCorrect, double loss) {
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

    public double getAccuracy() {
        if (nSamples == 0) {
            System.out.println("0.0, because no Predictions");
            return 0.0;
        } 
        
        double acc = (double) nCorrect / nSamples;
        return Math.round(acc * 1000.0) / 1000.0; // round at 3 decimal positions 
    }

    public double getBestAccuracy() {
        return Math.round(bestAccuracy * 1000.0) / 1000.0;
    }
    
    public void setBestAccuracy(double newBestAccuracy) {
        this.bestAccuracy = newBestAccuracy;
    }

    // Loss ==========================================================================

    public double getLoss() {
        return Math.round(loss * 1000.0) / 1000.0; 
    }
    
    public double getBestLoss() {
        return Math.round(bestLoss * 1000.0) / 1000.0; 
    }
    
    public void setBestLoss(double bestLoss) {
        this.bestLoss = bestLoss;
    }

    // ================================================================================

    public void reset() {
        nSamples = 0;
        nCorrect = 0;
        loss = 0;
    }
    
}

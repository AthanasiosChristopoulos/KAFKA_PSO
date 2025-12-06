package state;

public class Stats {

    private long nPredictions = 0;
    private long nCorrect = 0;
    private double loss = 0;

    private double bestAccuracy = 0;
    private double bestLoss = 1000000;

    public void addBatch(int nSamples, int nCorrectBatch, double loss) {
        nPredictions += nSamples;
        nCorrect += nCorrectBatch;
        loss += loss;

        double acc = nPredictions == 0 ? -1 : (double) nCorrect / nPredictions;
    }

    public long getNumPredictions() {
        return nPredictions;
    }

    public long getNumCorrect() {
        return nCorrect;
    }

    // Accuracy ==========================================================================

    public double getAccuracy() {
        if (nPredictions == 0) {
            System.out.println("0.0, because no Predictions");
            return 0.0;
        } 
        double acc = (double) nCorrect / nPredictions;
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
        return loss; 
    }
    
    public double getBestLoss() {
        return bestLoss;
    }
    
    public void setBestLoss(double bestLoss) {
        this.bestLoss = bestLoss;
    }

    // ================================================================================
    
    public void reset() {
        nPredictions = 0;
        nCorrect = 0;
        loss = 0;
    }
    
}

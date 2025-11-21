package pso;

public class Stats {

    private long nPredictions = 0;
    private long nCorrect = 0;

    private double bestAccuracy = 0;

    public void addBatch(int nSamples, int nCorrectBatch) {
        nPredictions += nSamples;
        nCorrect += nCorrectBatch;

        double acc = nPredictions == 0 ? -1 : (double) nCorrect / nPredictions;
    }

    public long getNumPredictions() {
        return nPredictions;
    }

    public long getNumCorrect() {
        return nCorrect;
    }

    public double getAccuracy() {
        if (nPredictions == 0) {
            System.out.println("AAAAA");
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

    public void reset() {
        nPredictions = 0;
        nCorrect = 0;
    }
    
}

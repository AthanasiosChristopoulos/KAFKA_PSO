package experimentation;

import java.util.*;

import utils.AccuracyPoint;

public final class CoordinatorMetrics {
    private final double elapsedSec;
    private final float globalBestAcc;
    private final float globalBestLoss;
    public final List<AccuracyPoint> accuracyValues;
    private final int dimensionality;
    private final double avgNetworkDelay;

    // ===========================================================================================

    public CoordinatorMetrics(double elapsedSec, float globalBestAcc, float globalBestLoss, 
            List<AccuracyPoint> accuracyValues, int dimensionality, double avgNetworkDelay) {
        this.elapsedSec = elapsedSec;
        this.globalBestAcc = globalBestAcc;
        this.globalBestLoss = globalBestLoss;
        this.accuracyValues = accuracyValues;
        this.dimensionality = dimensionality;
        this.avgNetworkDelay = avgNetworkDelay;
    }

    // ===========================================================================================

    public double getElapsedSec() { return elapsedSec; }
    public float getGlobalBestAcc() { return globalBestAcc; }
    public float getGlobalBestLoss() { return globalBestLoss; }
    public List<AccuracyPoint> getAccuracyValues() { return accuracyValues; }
    public int getDimensionality() { return dimensionality; }
    public double getAvgNetworkDelay() { return avgNetworkDelay; }

    // ===========================================================================================

    @Override
    public String toString() {
        return "CoordinatorMetrics{" +
                "elapsedSec=" + elapsedSec +
                ", globalBestAcc=" + globalBestAcc +
                ", globalBestLoss=" + globalBestLoss +
                '}';
    }
    
    // ===========================================================================================

    public void resetAccuracyValues() {
        if (accuracyValues != null) {
            accuracyValues.clear();
        }
    }

}
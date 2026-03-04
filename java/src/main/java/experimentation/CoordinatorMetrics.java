package experimentation;

import java.util.*;

public final class CoordinatorMetrics {
    private final double elapsedSec;
    private final float globalBestAcc;
    private final float globalBestLoss;

    public final List<Float> accuracyValues;
    
    public CoordinatorMetrics(double elapsedSec, float globalBestAcc, float globalBestLoss, List<Float> accuracyValues) {
        this.elapsedSec = elapsedSec;
        this.globalBestAcc = globalBestAcc;
        this.globalBestLoss = globalBestLoss;

        this.accuracyValues = accuracyValues;
    }

    public double getElapsedSec() { return elapsedSec; }
    public float getGlobalBestAcc() { return globalBestAcc; }
    public float getGlobalBestLoss() { return globalBestLoss; }

    @Override
    public String toString() {
        return "CoordinatorMetrics{" +
                "elapsedSec=" + elapsedSec +
                ", globalBestAcc=" + globalBestAcc +
                ", globalBestLoss=" + globalBestLoss +
                '}';
    }
}
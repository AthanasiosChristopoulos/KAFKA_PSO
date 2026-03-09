package utils;

public final class AccuracyPoint {
    private final double elapsedSec;
    private final float accuracy;

    public AccuracyPoint(double elapsedSec, float accuracy) {
        this.elapsedSec = elapsedSec;
        this.accuracy = accuracy;
    }

    public double getElapsedSec() {
        return elapsedSec;
    }

    public float getAccuracy() {
        return accuracy;
    }

    @Override
    public String toString() {
        return "AccuracyPoint{" +
                "elapsedSec=" + elapsedSec +
                ", accuracy=" + accuracy +
                '}';
    }
}
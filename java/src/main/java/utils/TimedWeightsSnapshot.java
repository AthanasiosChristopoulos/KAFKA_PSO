package utils;

public final class TimedWeightsSnapshot {
    public final float[] weights;
    public final long timeMs;

    public TimedWeightsSnapshot(float[] weights, long timeMs) {
        this.weights = weights;
        this.timeMs = timeMs;
    }
}
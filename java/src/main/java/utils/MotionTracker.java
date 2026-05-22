package utils;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

public final class MotionTracker {
    
    private final int maxWindow;
    private final Deque<TimedWeightsSnapshot> window = new ArrayDeque<>();

    private float[] currentVelocity = null;
    private long currentVelocityTime = -1L;

    private float[] previousVelocity = null;
    private long previousVelocityTime = -1L;

    // ==============================================================================

    public MotionTracker(int maxWindow) {
        this.maxWindow = Math.max(2, maxWindow);
    }

    // ==============================================================================

    public void addSnapshot(float[] weights, long timeMs) {
        window.addLast(new TimedWeightsSnapshot(Arrays.copyOf(weights, weights.length), timeMs));
        while (window.size() > maxWindow) {
            window.removeFirst();
        }
    }

    // ==============================================================================

    public TimedWeightsSnapshot getLatestSnapshot() {
        if (window.isEmpty()) {
            return null;
        }
        return window.peekLast();  
    }
    
    // ==============================================================================

    public boolean hasVelocity() {
        return window.size() >= 2;
    }

    // ==============================================================================

    public boolean hasAcceleration() {
        return currentVelocity != null && previousVelocity != null;
    }

    // ==============================================================================

    public float[] estimateWindowVelocity() {

        if (window.size() < 2) return null;

        TimedWeightsSnapshot first = window.peekFirst();
        TimedWeightsSnapshot last = window.peekLast();

        long dt = Math.max(1L, last.timeMs - first.timeMs);
        float[] vel = new float[last.weights.length];

        for (int i = 0; i < vel.length; i++) {
            vel[i] = (float)(((double) last.weights[i] - (double) first.weights[i]) / (double) dt);
        }

        previousVelocity = currentVelocity;
        previousVelocityTime = currentVelocityTime;

        currentVelocity = vel;
        currentVelocityTime = last.timeMs;

        return vel;
    }

    // ==============================================================================

    public float[] estimateAcceleration() {
        if (currentVelocity == null || previousVelocity == null) return null;

        long dt = Math.max(1L, currentVelocityTime - previousVelocityTime);
        float[] acc = new float[currentVelocity.length];

        for (int i = 0; i < acc.length; i++) {
            acc[i] = (float)(((double) currentVelocity[i] - (double) previousVelocity[i]) / (double) dt);
        }

        return acc;
    }

    // ==============================================================================

    public float[] getCurrentVelocity() {
        return currentVelocity;
    }

    // ==============================================================================

    public long getCurrentVelocityTime() {
        return currentVelocityTime;
    }
}
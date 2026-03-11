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
        return window.peekLast();   // most recent snapshot
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

    public float[] estimateWindowVelocity_v2() {
        if (window.size() < 2) return null;

        TimedWeightsSnapshot[] snaps = window.toArray(new TimedWeightsSnapshot[0]);
        int m = snaps.length;
        int d = snaps[0].weights.length;

        float[] vel = new float[d];

        // Mean time
        double meanT = 0.0;
        for (TimedWeightsSnapshot s : snaps) {
            meanT += s.timeMs;
        }
        meanT /= m;

        // Denominator of slope
        double denom = 0.0;
        for (TimedWeightsSnapshot s : snaps) {
            double dt = s.timeMs - meanT;
            denom += dt * dt;
        }

        if (denom < 1e-12) {
            Arrays.fill(vel, 0f);
        } else {
            for (int j = 0; j < d; j++) {
                double meanX = 0.0;
                for (TimedWeightsSnapshot s : snaps) {
                    meanX += s.weights[j];
                }
                meanX /= m;

                double numer = 0.0;
                for (TimedWeightsSnapshot s : snaps) {
                    double dt = s.timeMs - meanT;
                    double dx = s.weights[j] - meanX;
                    numer += dt * dx;
                }

                vel[j] = (float)(numer / denom);
            }
        }

        previousVelocity = currentVelocity;
        previousVelocityTime = currentVelocityTime;

        currentVelocity = vel;
        currentVelocityTime = snaps[m - 1].timeMs;

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
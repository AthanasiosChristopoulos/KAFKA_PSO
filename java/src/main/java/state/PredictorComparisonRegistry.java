package state;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

public final class PredictorComparisonRegistry {

    public enum Kind {
        PBEST,
        MONITORING
    }

    private static final class Stats2 {
        final AtomicLong totalComparisons = new AtomicLong(0);
        final AtomicLong staticWins = new AtomicLong(0);
        final AtomicLong linearWins = new AtomicLong(0);
        final AtomicLong ties = new AtomicLong(0);

        final AtomicLong linearSatisfiesGoodPredictor = new AtomicLong(0);
        final AtomicLong linearViolatesGoodPredictor = new AtomicLong(0);

        final DoubleAdder staticErrorSum = new DoubleAdder();
        final DoubleAdder linearErrorSum = new DoubleAdder();
    }

    private static final class Stats3 {
        final AtomicLong totalComparisons = new AtomicLong(0);

        final AtomicLong staticWins = new AtomicLong(0);
        final AtomicLong linearWins = new AtomicLong(0);
        final AtomicLong psoVelocityWins = new AtomicLong(0);
        final AtomicLong ties = new AtomicLong(0);

        final AtomicLong linearSatisfiesGoodPredictor = new AtomicLong(0);
        final AtomicLong linearViolatesGoodPredictor = new AtomicLong(0);

        final AtomicLong psoVelocitySatisfiesGoodPredictor = new AtomicLong(0);
        final AtomicLong psoVelocityViolatesGoodPredictor = new AtomicLong(0);

        final DoubleAdder staticErrorSum = new DoubleAdder();
        final DoubleAdder linearErrorSum = new DoubleAdder();
        final DoubleAdder psoVelocityErrorSum = new DoubleAdder();
    }

    private static final Stats2 PBEST_STATS = new Stats2();
    private static final Stats3 MONITORING_STATS = new Stats3();

    private PredictorComparisonRegistry() {}

    public static void record(Kind kind, double staticErr, double linearErr) {
        if (kind != Kind.PBEST) {
            throw new IllegalArgumentException("2-way record() is only for PBEST");
        }

        Stats2 s = PBEST_STATS;

        s.totalComparisons.incrementAndGet();
        s.staticErrorSum.add(staticErr);
        s.linearErrorSum.add(linearErr);

        if (linearErr <= staticErr) {
            s.linearSatisfiesGoodPredictor.incrementAndGet();
        } else {
            s.linearViolatesGoodPredictor.incrementAndGet();
        }

        double eps = 1e-12;
        if (Math.abs(linearErr - staticErr) <= eps) {
            s.ties.incrementAndGet();
        } else if (linearErr < staticErr) {
            s.linearWins.incrementAndGet();
        } else {
            s.staticWins.incrementAndGet();
        }
    }

    public static void recordMonitoring(double staticErr, double linearErr, double psoVelErr) {
        Stats3 s = MONITORING_STATS;

        s.totalComparisons.incrementAndGet();
        s.staticErrorSum.add(staticErr);
        s.linearErrorSum.add(linearErr);
        s.psoVelocityErrorSum.add(psoVelErr);

        if (linearErr <= staticErr) {
            s.linearSatisfiesGoodPredictor.incrementAndGet();
        } else {
            s.linearViolatesGoodPredictor.incrementAndGet();
        }

        if (psoVelErr <= staticErr) {
            s.psoVelocitySatisfiesGoodPredictor.incrementAndGet();
        } else {
            s.psoVelocityViolatesGoodPredictor.incrementAndGet();
        }

        double min = Math.min(staticErr, Math.min(linearErr, psoVelErr));
        double eps = 1e-12;

        int winners = 0;
        if (Math.abs(staticErr - min) <= eps) winners++;
        if (Math.abs(linearErr - min) <= eps) winners++;
        if (Math.abs(psoVelErr - min) <= eps) winners++;

        if (winners > 1) {
            s.ties.incrementAndGet();
        } else if (Math.abs(staticErr - min) <= eps) {
            s.staticWins.incrementAndGet();
        } else if (Math.abs(linearErr - min) <= eps) {
            s.linearWins.incrementAndGet();
        } else {
            s.psoVelocityWins.incrementAndGet();
        }
    }

    private static String summaryForPBest() {
        Stats2 s = PBEST_STATS;
        long n = s.totalComparisons.get();
        double avgStatic = (n == 0) ? Double.NaN : s.staticErrorSum.sum() / n;
        double avgLinear = (n == 0) ? Double.NaN : s.linearErrorSum.sum() / n;

        String winner;
        if (Double.isNaN(avgStatic) || Double.isNaN(avgLinear)) {
            winner = "NO_COMPARISONS";
        } else if (avgLinear < avgStatic) {
            winner = "LINEAR_GROWTH";
        } else if (avgStatic < avgLinear) {
            winner = "STATIC";
        } else {
            winner = "TIE";
        }

        return "\n========== PBEST Predictor Comparison ==========\n" +
               "totalComparisons = " + n + "\n" +
               "staticWins       = " + s.staticWins.get() + "\n" +
               "linearWins       = " + s.linearWins.get() + "\n" +
               "ties             = " + s.ties.get() + "\n" +
               "linear<=static   = " + s.linearSatisfiesGoodPredictor.get() + "\n" +
               "linear>static    = " + s.linearViolatesGoodPredictor.get() + "\n" +
               "avgStaticError   = " + avgStatic + "\n" +
               "avgLinearError   = " + avgLinear + "\n" +
               "WINNER           = " + winner + "\n" +
               "================================================";
    }

    private static String summaryForMonitoring() {
        Stats3 s = MONITORING_STATS;
        long n = s.totalComparisons.get();

        double avgStatic = (n == 0) ? Double.NaN : s.staticErrorSum.sum() / n;
        double avgLinear = (n == 0) ? Double.NaN : s.linearErrorSum.sum() / n;
        double avgPsoVel = (n == 0) ? Double.NaN : s.psoVelocityErrorSum.sum() / n;

        String winner;
        if (Double.isNaN(avgStatic) || Double.isNaN(avgLinear) || Double.isNaN(avgPsoVel)) {
            winner = "NO_COMPARISONS";
        } else {
            double min = Math.min(avgStatic, Math.min(avgLinear, avgPsoVel));
            if (min == avgPsoVel) winner = "PSO_VELOCITY";
            else if (min == avgLinear) winner = "LINEAR_GROWTH";
            else winner = "STATIC";
        }

        return "\n========== MONITORING Predictor Comparison ==========\n" +
               "totalComparisons     = " + n + "\n" +
               "staticWins           = " + s.staticWins.get() + "\n" +
               "linearWins           = " + s.linearWins.get() + "\n" +
               "psoVelocityWins      = " + s.psoVelocityWins.get() + "\n" +
               "ties                 = " + s.ties.get() + "\n" +
               "linear<=static       = " + s.linearSatisfiesGoodPredictor.get() + "\n" +
               "linear>static        = " + s.linearViolatesGoodPredictor.get() + "\n" +
               "psoVel<=static       = " + s.psoVelocitySatisfiesGoodPredictor.get() + "\n" +
               "psoVel>static        = " + s.psoVelocityViolatesGoodPredictor.get() + "\n" +
               "avgStaticError       = " + avgStatic + "\n" +
               "avgLinearError       = " + avgLinear + "\n" +
               "avgPsoVelocityError  = " + avgPsoVel + "\n" +
               "WINNER               = " + winner + "\n" +
               "====================================================";
    }

    public static String summary() {
        return summaryForPBest() + "\n" + summaryForMonitoring();
    }
}
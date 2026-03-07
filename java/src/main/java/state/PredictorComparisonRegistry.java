package state;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

public final class PredictorComparisonRegistry {

    public enum Kind {
        PBEST,
        MONITORING
    }

    private static final class Stats {
        final AtomicLong totalComparisons = new AtomicLong(0);
        final AtomicLong staticWins = new AtomicLong(0);
        final AtomicLong linearWins = new AtomicLong(0);
        final AtomicLong ties = new AtomicLong(0);

        final AtomicLong linearSatisfiesGoodPredictor = new AtomicLong(0);
        final AtomicLong linearViolatesGoodPredictor = new AtomicLong(0);

        final DoubleAdder staticErrorSum = new DoubleAdder();
        final DoubleAdder linearErrorSum = new DoubleAdder();
    }

    private static final Stats PBEST_STATS = new Stats();
    private static final Stats MONITORING_STATS = new Stats();

    private PredictorComparisonRegistry() {}

    private static Stats statsFor(Kind kind) {
        return (kind == Kind.PBEST) ? PBEST_STATS : MONITORING_STATS;
    }

    public static void record(Kind kind, double staticErr, double linearErr) {
        Stats s = statsFor(kind);

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

    private static String summaryFor(String title, Stats s) {
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

        return "\n========== " + title + " Predictor Comparison ==========\n" +
               "totalComparisons = " + n + "\n" +
               "staticWins       = " + s.staticWins.get() + "\n" +
               "linearWins       = " + s.linearWins.get() + "\n" +
               "ties             = " + s.ties.get() + "\n" +
               "linear<=static   = " + s.linearSatisfiesGoodPredictor.get() + "\n" +
               "linear>static    = " + s.linearViolatesGoodPredictor.get() + "\n" +
               "avgStaticError   = " + avgStatic + "\n" +
               "avgLinearError   = " + avgLinear + "\n" +
               "WINNER           = " + winner + "\n" +
               "====================================================";
    }

    public static String summary() {
        return summaryFor("PBEST", PBEST_STATS) +
               "\n" +
               summaryFor("MONITORING", MONITORING_STATS);
    }
}
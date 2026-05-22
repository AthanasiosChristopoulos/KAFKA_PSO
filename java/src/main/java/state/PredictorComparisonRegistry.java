package state;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

public final class PredictorComparisonRegistry {

    public enum Kind {
        PBEST,
        MONITORING
    }

    private static final class StatsMonitoring {
        final AtomicLong totalComparisons = new AtomicLong(0);

        final AtomicLong staticWins = new AtomicLong(0);
        final AtomicLong linearWins = new AtomicLong(0);
        final AtomicLong psoVelocityWins = new AtomicLong(0);
        final AtomicLong observedVelocityWins = new AtomicLong(0);
        final AtomicLong vaWins = new AtomicLong(0);
        final AtomicLong ties = new AtomicLong(0);

        final DoubleAdder staticErrorSum = new DoubleAdder();
        final DoubleAdder linearErrorSum = new DoubleAdder();
        final DoubleAdder psoVelocityErrorSum = new DoubleAdder();
        final DoubleAdder observedVelocityErrorSum = new DoubleAdder();
        final DoubleAdder vaErrorSum = new DoubleAdder();
    }

    private static final StatsMonitoring PBEST_STATS = new StatsMonitoring();
    private static final StatsMonitoring MONITORING_STATS = new StatsMonitoring();

    private PredictorComparisonRegistry() {}

    // =====================================================================

    public static void recordPBest(Kind kind, double staticErr, double linearErr, double psoVelErr,
            double observedVelErr, double vaErr) {

        if (kind != Kind.PBEST) {
            throw new IllegalArgumentException("record() is only for PBEST");
        }

        StatsMonitoring s = PBEST_STATS;

        s.totalComparisons.incrementAndGet();
        s.staticErrorSum.add(staticErr);
        s.linearErrorSum.add(linearErr);
        s.psoVelocityErrorSum.add(psoVelErr);
        s.observedVelocityErrorSum.add(observedVelErr);
        s.vaErrorSum.add(vaErr);

        double min = Math.min(staticErr,
            Math.min(linearErr, Math.min(psoVelErr, Math.min(observedVelErr, vaErr)))
        );

        double eps = 1e-12;
        int winners = 0;

        if (Math.abs(staticErr - min) <= eps) winners++;
        if (Math.abs(linearErr - min) <= eps) winners++;
        if (Math.abs(psoVelErr - min) <= eps) winners++;
        if (Math.abs(observedVelErr - min) <= eps) winners++;
        if (Math.abs(vaErr - min) <= eps) winners++;

        if (winners > 1) {
            s.ties.incrementAndGet();
        } else if (Math.abs(staticErr - min) <= eps) {
            s.staticWins.incrementAndGet();
        } else if (Math.abs(linearErr - min) <= eps) {
            s.linearWins.incrementAndGet();
        } else if (Math.abs(psoVelErr - min) <= eps) {
            s.psoVelocityWins.incrementAndGet();
        } else if (Math.abs(observedVelErr - min) <= eps) {
            s.observedVelocityWins.incrementAndGet();
        } else {
            s.vaWins.incrementAndGet();
        }
    }

    // =====================================================================

    public static void recordMonitoring(Kind kind, double staticErr, double linearErr, double psoVelErr,
            double observedVelErr, double vaErr) {
        if (kind != Kind.MONITORING) {
            throw new IllegalArgumentException("recordMonitoring() is only for MONITORING");
        }
        StatsMonitoring s = MONITORING_STATS;

        s.totalComparisons.incrementAndGet();
        s.staticErrorSum.add(staticErr);
        s.linearErrorSum.add(linearErr);
        s.psoVelocityErrorSum.add(psoVelErr);
        s.observedVelocityErrorSum.add(observedVelErr);
        s.vaErrorSum.add(vaErr);

        double min = Math.min(staticErr,
            Math.min(linearErr, Math.min(psoVelErr, Math.min(observedVelErr, vaErr)))
        );

        double eps = 1e-12;
        int winners = 0;

        if (Math.abs(staticErr - min) <= eps) winners++;
        if (Math.abs(linearErr - min) <= eps) winners++;
        if (Math.abs(psoVelErr - min) <= eps) winners++;
        if (Math.abs(observedVelErr - min) <= eps) winners++;
        if (Math.abs(vaErr - min) <= eps) winners++;

        if (winners > 1) {
            s.ties.incrementAndGet();
        } else if (Math.abs(staticErr - min) <= eps) {
            s.staticWins.incrementAndGet();
        } else if (Math.abs(linearErr - min) <= eps) {
            s.linearWins.incrementAndGet();
        } else if (Math.abs(psoVelErr - min) <= eps) {
            s.psoVelocityWins.incrementAndGet();
        } else if (Math.abs(observedVelErr - min) <= eps) {
            s.observedVelocityWins.incrementAndGet();
        } else {
            s.vaWins.incrementAndGet();
        }
    }

    // =======================================================================================

    private static String summaryForPBest() {

        StatsMonitoring s = PBEST_STATS;
        long n = s.totalComparisons.get();

        double avgStatic = (n == 0) ? Double.NaN : s.staticErrorSum.sum() / n;
        double avgLinear = (n == 0) ? Double.NaN : s.linearErrorSum.sum() / n;
        double avgPsoVel = (n == 0) ? Double.NaN : s.psoVelocityErrorSum.sum() / n;
        double avgObsVel = (n == 0) ? Double.NaN : s.observedVelocityErrorSum.sum() / n;
        double avgVA = (n == 0) ? Double.NaN : s.vaErrorSum.sum() / n;

        String winner;
        if (Double.isNaN(avgStatic) || Double.isNaN(avgLinear) || Double.isNaN(avgPsoVel)
                || Double.isNaN(avgObsVel) || Double.isNaN(avgVA)) {
            winner = "NO_COMPARISONS";
        } else {
            double min = Math.min(avgStatic,
                Math.min(avgLinear, Math.min(avgPsoVel, Math.min(avgObsVel, avgVA)))
            );

            if (min == avgVA) winner = "VELOCITY_ACCELERATION";
            else if (min == avgObsVel) winner = "OBSERVED_VELOCITY";
            else if (min == avgPsoVel) winner = "PSO_VELOCITY";
            else if (min == avgLinear) winner = "LINEAR_GROWTH";
            else winner = "STATIC";
        }

        return "\n========== PBEST Predictor Comparison ==========\n" +
            "totalComparisons       = " + n + "\n" +
            "staticWins             = " + s.staticWins.get() + "\n" +
            "linearWins             = " + s.linearWins.get() + "\n" +
            "psoVelocityWins        = " + s.psoVelocityWins.get() + "\n" +
            "observedVelocityWins   = " + s.observedVelocityWins.get() + "\n" +
            "vaWins                 = " + s.vaWins.get() + "\n" +
            "ties                   = " + s.ties.get() + "\n" +
            "avgStaticError         = " + avgStatic + "\n" +
            "avgLinearError         = " + avgLinear + "\n" +
            "avgPsoVelocityError    = " + avgPsoVel + "\n" +
            "avgObservedVelocityErr = " + avgObsVel + "\n" +
            "avgVAError             = " + avgVA + "\n" +
            "WINNER                 = " + winner + "\n" +
            "================================================";
    }

    // =======================================================================================
    
    private static String summaryForMonitoring() {

        StatsMonitoring s = MONITORING_STATS;
        long n = s.totalComparisons.get();

        double avgStatic = (n == 0) ? Double.NaN : s.staticErrorSum.sum() / n;
        double avgLinear = (n == 0) ? Double.NaN : s.linearErrorSum.sum() / n;
        double avgPsoVel = (n == 0) ? Double.NaN : s.psoVelocityErrorSum.sum() / n;
        double avgObsVel = (n == 0) ? Double.NaN : s.observedVelocityErrorSum.sum() / n;
        double avgVA = (n == 0) ? Double.NaN : s.vaErrorSum.sum() / n;

        String winner;
        if (Double.isNaN(avgStatic) || Double.isNaN(avgLinear) || Double.isNaN(avgPsoVel)
                || Double.isNaN(avgObsVel) || Double.isNaN(avgVA)) {
            winner = "NO_COMPARISONS";
        } else {
            double min = Math.min(avgStatic,
                Math.min(avgLinear, Math.min(avgPsoVel, Math.min(avgObsVel, avgVA)))
            );

            if (min == avgVA) winner = "VELOCITY_ACCELERATION";
            else if (min == avgObsVel) winner = "OBSERVED_VELOCITY";
            else if (min == avgPsoVel) winner = "PSO_VELOCITY";
            else if (min == avgLinear) winner = "LINEAR_GROWTH";
            else winner = "STATIC";
        }

        return "\n========== MONITORING Predictor Comparison ==========\n" +
            "totalComparisons       = " + n + "\n" +
            "staticWins             = " + s.staticWins.get() + "\n" +
            "linearWins             = " + s.linearWins.get() + "\n" +
            "psoVelocityWins        = " + s.psoVelocityWins.get() + "\n" +
            "observedVelocityWins   = " + s.observedVelocityWins.get() + "\n" +
            "vaWins                 = " + s.vaWins.get() + "\n" +
            "ties                   = " + s.ties.get() + "\n" +
            "avgStaticError         = " + avgStatic + "\n" +
            "avgLinearError         = " + avgLinear + "\n" +
            "avgPsoVelocityError    = " + avgPsoVel + "\n" +
            "avgObservedVelocityErr = " + avgObsVel + "\n" +
            "avgVAError             = " + avgVA + "\n" +
            "WINNER                 = " + winner + "\n" +
            "====================================================";
    }

    // ==================================================================================
    
    public static String summary() {
        return summaryForPBest() + "\n" + summaryForMonitoring();

    }
}
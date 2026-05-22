package experimentation;

import java.util.Collections;
import java.util.List;

import utils.AccuracyPoint;

public final class ExperimentResult {
    private final int nWorkers;
    private final double totalElapsedSec;
    private final CoordinatorMetrics coordinator;
    private final List<WorkerMetrics> workers;

    // ===============================================================================

    public ExperimentResult(int nWorkers, double totalElapsedSec, CoordinatorMetrics coordinator, List<WorkerMetrics> workers) {
        this.nWorkers = nWorkers;
        this.totalElapsedSec = totalElapsedSec;
        this.coordinator = coordinator;
        this.workers = workers == null ? Collections.emptyList() : Collections.unmodifiableList(workers);
    }

    // ===============================================================================

    public int getNWorkers() { return nWorkers; }
    public double getTotalElapsedSec() { return totalElapsedSec; }
    public CoordinatorMetrics getCoordinator() { return coordinator; }
    public List<WorkerMetrics> getWorkers() { return workers; }

    // ===============================================================================

    public double lastWorkerElapsedSec() {
        double max = Double.NaN;
        
        for (WorkerMetrics wm : workers) {
            if (wm == null) continue;
            
            double t = wm.getElapsedSec();

            if (Double.isNaN(max) || t > max) {
                max = t;
            }
        }

        return max;
    }
    
    // ===============================================================================

    public List<AccuracyPoint> getAccuracyValues() {
        return coordinator.accuracyValues;
    }
    
    // ===============================================================================

    public double getAvgNetworkDelay() {
        return coordinator.getAvgNetworkDelay();
    }

    // ===============================================================================

    public void resetAccuracyValues() {
        coordinator.resetAccuracyValues();
    }
    
    // ===============================================================================

    public long sumMessagesSent() {
        return workers.stream()
                .mapToLong(WorkerMetrics::getTOTAL_MESSAGES_SENT)
                .sum();
    }

    // ===============================================================================
    
    public long sumPBestMessagesSent() {
        return workers.stream()
                .mapToLong(WorkerMetrics::getTOTAL_MESSAGES_SENT_PBEST)
                .sum();
    }    

    // ===============================================================================

    public long sumCurrentWeightsMessagesSent() {
        return workers.stream()
                .mapToLong(WorkerMetrics::getTOTAL_MESSAGES_SENT_CURRENT_WEIGHTS)
                .sum();
    }
    // ===============================================================================

    public long sumBytesSent() {
        return workers.stream()
                .mapToLong(WorkerMetrics::getTOTAL_BYTES_SENT)
                .sum();
    }

    // ===============================================================================

    public int getDimensionality() {
        return coordinator.getDimensionality();
    }
    
    // ===============================================================================

    @Override
    public String toString() {
        return "ExperimentResult{" +
                "nWorkers=" + nWorkers +
                ", totalElapsedSec=" + totalElapsedSec +
                ", coordinator=" + coordinator +
                ", workers=" + workers +
                '}';
    }
}
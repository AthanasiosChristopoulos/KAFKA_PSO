package experimentation;

import java.util.Collections;
import java.util.List;

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
            if (Double.isNaN(max) || t > max) max = t;
        }
        return max;
    }

    // ===============================================================================

    public long maxMessagesSent() {
        return workers.stream()
                .mapToLong(WorkerMetrics::getTOTAL_MESSAGES_SENT)
                .max()
                .orElse(0L);
    }

    // ===============================================================================

    public long maxBytesSent() {
        return workers.stream()
                .mapToLong(WorkerMetrics::getTOTAL_BYTES_SENT)
                .max()
                .orElse(0L);
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
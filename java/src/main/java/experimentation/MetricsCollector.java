package experimentation;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;

public class MetricsCollector {

    private final CountDownLatch workersDone;
    private final Map<Integer, WorkerMetrics> workerMetrics = new ConcurrentHashMap<>();
    private volatile CoordinatorMetrics coordinatorMetrics;

    public MetricsCollector(int nWorkers) {
        this.workersDone = new CountDownLatch(nWorkers);
    }

    public void reportWorkerDone(WorkerMetrics m) {
        workerMetrics.put(m.getWorkerId(), m);
        workersDone.countDown();
    }

    public void reportCoordinatorDone(CoordinatorMetrics m) {
        this.coordinatorMetrics = m;
    }

    public void awaitWorkers() throws InterruptedException {
        workersDone.await();
    }

    public ExperimentResult buildResult(int nWorkers, double totalElapsedSec) {
        List<WorkerMetrics> list = workerMetrics.values().stream()
                .sorted(Comparator.comparingInt(WorkerMetrics::getWorkerId))
                .toList();
        return new ExperimentResult(nWorkers, totalElapsedSec, coordinatorMetrics, list);
    }
}
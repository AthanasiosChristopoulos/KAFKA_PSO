package experimentation;

import pso.*;
import utils.*;

public class SimulationRunnerCoordinator {

    public static ExperimentResult runOnce(Config cfg) throws Exception {
        int numWorkers = cfg.N_WORKERS;

        var baseStateDir = java.nio.file.Path.of(cfg.KAFKA_TMP_DIR + "/run-" + System.currentTimeMillis());
        java.nio.file.Files.createDirectories(baseStateDir);

        MetricsCollector collector = new MetricsCollector(numWorkers);

        long start = System.nanoTime();

        Coordinator coordinator = new Coordinator(collector);
        Thread coordinatorThread = new Thread(coordinator, "coordinator");
        coordinatorThread.start();

        double totalElapsedSec = (System.nanoTime() - start) / 1_000_000_000.0;

        coordinatorThread.join();

        return collector.buildResult(numWorkers, totalElapsedSec);
    }
}
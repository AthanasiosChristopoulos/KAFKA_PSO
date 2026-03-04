package experimentation;

import pso.*;
import utils.*;
import java.util.*;

public class SimulationRunner {

    public static ExperimentResult runOnce(Config cfg) throws Exception {
        int numWorkers = cfg.N_WORKERS;

        // IMPORTANT for Kafka Streams: unique state dir / application.id per run
        var baseStateDir = java.nio.file.Path.of(cfg.KAFKA_TMP_DIR + "/run-" + System.currentTimeMillis());
        java.nio.file.Files.createDirectories(baseStateDir);

        MetricsCollector collector = new MetricsCollector(numWorkers);

        long start = System.nanoTime();

        Coordinator coordinator = new Coordinator(collector); // pass collector
        Thread coordinatorThread = new Thread(coordinator, "coordinator");
        coordinatorThread.start();

        List<Thread> workerThreads = new ArrayList<>();
        for (int i = 0; i < numWorkers; i++) {
            Worker worker = new Worker(i, collector); // pass collector
            Thread t = new Thread(worker, "worker-thread-" + i);
            t.start();
            workerThreads.add(t);
        }

        for (Thread t : workerThreads) t.join();

        double totalElapsedSec = (System.nanoTime() - start) / 1_000_000_000.0;

        coordinatorThread.join();
        collector.awaitWorkers();

        return collector.buildResult(numWorkers, totalElapsedSec);
    }
}
package experimentation;
import utils.*;
import state.*;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.Duration;

public class Experimentation {

    private static final Config cfg = Config.getInstance();

    public static void main(String[] args) throws Exception {

        List<Integer> workersList = List.of(2, 4, 6);
        Path resultsDir = createUniqueResultsDir("experimental_results_v1");
        Path csvPath = resultsDir.resolve("results.csv");

        try (BufferedWriter w = Files.newBufferedWriter(
                csvPath,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.WRITE
        )) {
            w.write("N_WORKERS,TOTAL_ELAPSED,COORD_ELAPSED,LAST_WORKER_ELAPSED,GBEST_ACC,GBEST_LOSS\n");

            for (int n : workersList) {

                cfg.refreshRunId();

                System.out.println("===============================================================================================");
                System.out.println("N_WORKERS: " + n);
                System.out.println("New RUN_ID: " + cfg.RUN_ID);
                System.out.println("===============================================================================================");

                // =================================================================================================

                List<String> topics;
                if (cfg.FULLY_INFORMED || cfg.ENABLE_NEIGHBORHOODS) {
                    topics = List.of(cfg.PBEST_WEIGHTS_TOPIC);
                } else {
                    topics = List.of(cfg.GLOBAL_WEIGHTS_TOPIC);
                }

                int partitions = 1; // or nWorkers
                short rf = 1; // in your docker broker, replication factor is usually 1
                // KafkaTopicManager.recreateTopics("localhost:9092", topics, partitions, rf);
                KafkaTopicManager.recreateTopics(
                        "localhost:9092",
                        topics,
                        1,                 // partitions (or nWorkers)
                        (short) 1,         // replication factor
                        Duration.ofSeconds(30),
                        Duration.ofSeconds(30)
                );
                // =================================================================================================

                Config cfg = Config.getInstance();
                cfg.N_WORKERS = n; // or better: create a Config copy per run

                ExperimentResult r = SimulationRunner.runOnce(cfg);

                double coordElapsed = r.getCoordinator() != null ? r.getCoordinator().getElapsedSec() : Double.NaN;

                w.write(String.format(
                        "%.3f,%.3f,%.3f,%.6f,%.6f\n",
                        //r.getWorkers(),
                        r.getTotalElapsedSec(),
                        coordElapsed,
                        r.lastWorkerElapsedSec(),
                        r.getCoordinator() != null ? r.getCoordinator().getGlobalBestAcc() : Double.NaN,
                        r.getCoordinator() != null ? r.getCoordinator().getGlobalBestLoss() : Double.NaN
                ));
                w.flush();
                System.out.println("===============================================================================================");
                System.out.println("End of experiment with N_WORKERS: " + n);
                System.out.println("===============================================================================================");
            }
        }
    }

    private static Path createUniqueResultsDir(String baseDirName) throws IOException {
        Files.createDirectories(Path.of(baseDirName));

        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        int rnd = ThreadLocalRandom.current().nextInt(1000, 10000); // 4 digits

        // Example: experimental_results_v1/run_20260224_154233_4821/
        Path runDir = Path.of(baseDirName, "run_" + ts + "_" + rnd);

        // Make sure it’s unique even if insanely unlucky
        int attempt = 0;
        while (Files.exists(runDir)) {
            rnd = ThreadLocalRandom.current().nextInt(1000, 10000);
            runDir = Path.of(baseDirName, "run_" + ts + "_" + rnd);
            if (++attempt > 50) throw new IOException("Could not create unique run directory under " + baseDirName);
        }

        Files.createDirectories(runDir);
        return runDir;
    }
}
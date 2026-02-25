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
    private static String bootstrap = "localhost:9092";
    private static final float THRESH_CENTER = 0.055f;
    private static final float MIN_FLOOR = 0.001f;
        
    public static void main(String[] args) throws Exception {

        if(cfg.EXPERIMENTATION_MODE.equals("N_WORKERS")) {
            List<Integer> workersList = List.of(2, 4, 6);
            // Path csvPath = createUniqueCsvPath("experimental_results_v1", "results");
            Path dir = Path.of("experimental_results_v1");
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_n_workers.csv");

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write("N_WORKERS,TOTAL_ELAPSED,COORD_ELAPSED,LAST_WORKER_ELAPSED,GBEST_ACC,GBEST_LOSS,TOTAL_MESSAGES_SENT,TOTAL_BYTES_SENT,LOSS_THRESHOLD_DIFF\n");

                for (int n : workersList) {

                    cfg.refreshRunId();
                    cfg.N_WORKERS = n;
                    CoordinatorControl.getInstance().resetForNewRun(n);

                    System.out.println("===============================================================================================");
                    System.out.println("N_WORKERS: " + n);
                    System.out.println("New RUN_ID: " + cfg.RUN_ID);
                    System.out.println("===============================================================================================");

                    // =================================================================================================
                    // Restart the Kafka Parititions

                    List<String> topics;
                    if (cfg.FULLY_INFORMED || cfg.ENABLE_NEIGHBORHOODS) {
                        topics = List.of(cfg.PBEST_WEIGHTS_TOPIC);
                    } else {
                        topics = List.of(cfg.GLOBAL_WEIGHTS_TOPIC);
                    }

                    KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                    // =================================================================================================

                    ExperimentResult r = SimulationRunner.runOnce(cfg);

                    double coordElapsed = r.getCoordinator() != null ? r.getCoordinator().getElapsedSec() : Double.NaN;

                    w.write(String.format(
                            "%d,%.3f,%.3f,%.3f,%.6f,%.6f,%d,%d,%f\n",
                            n,
                            r.getTotalElapsedSec(),
                            coordElapsed,
                            r.lastWorkerElapsedSec(),
                            r.getCoordinator() != null ? r.getCoordinator().getGlobalBestAcc() : Double.NaN,
                            r.getCoordinator() != null ? r.getCoordinator().getGlobalBestLoss() : Double.NaN,
                            r.maxMessagesSent(),
                            r.maxBytesSent(),
                            cfg.LOSS_THRESHOLD_MAX - cfg.LOSS_THRESHOLD_MIN
                    ));

                    w.flush();
                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with N_WORKERS: " + n);
                    System.out.println("===============================================================================================");

                }
            } 
        
        // ===========================================================================================================================================
        // ===========================================================================================================================================

        } else if(cfg.EXPERIMENTATION_MODE.equals("THRESHOLD")){
                         
            List<Float> theshold_offset_list = List.of(0.01f, 0.02f, 0.03f);

            // Path csvPath = createUniqueCsvPath("experimental_results_v1", "results");
            Path dir = Path.of("experimental_results_v2");
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_threshold.csv");

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write("N_WORKERS,TOTAL_ELAPSED,COORD_ELAPSED,LAST_WORKER_ELAPSED,GBEST_ACC,GBEST_LOSS,TOTAL_MESSAGES_SENT,TOTAL_BYTES_SENT,LOSS_THRESHOLD_DIFF\n");

                for (float theshold_offset : theshold_offset_list) {

                    cfg.refreshRunId();
                    cfg.LOSS_THRESHOLD_MIN = cfg.LOSS_THRESHOLD_MIN + theshold_offset;
                    cfg.LOSS_THRESHOLD_MAX = cfg.LOSS_THRESHOLD_MAX + theshold_offset;
                    CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);

                    System.out.println("===============================================================================================");
                    System.out.println("LOSS_THRESHOLD_MIN=" + cfg.LOSS_THRESHOLD_MIN + ", LOSS_THRESHOLD_MAX=" + cfg.LOSS_THRESHOLD_MAX + ", DIFF=" + theshold_offset);
                    System.out.println("New RUN_ID: " + cfg.RUN_ID);
                    System.out.println("===============================================================================================");

                    // =================================================================================================
                    // Restart the Kafka Parititions

                    List<String> topics;
                    if (cfg.FULLY_INFORMED || cfg.ENABLE_NEIGHBORHOODS) {
                        topics = List.of(cfg.PBEST_WEIGHTS_TOPIC);
                    } else {
                        topics = List.of(cfg.GLOBAL_WEIGHTS_TOPIC);
                    }

                    KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                    // =================================================================================================

                    ExperimentResult r = SimulationRunner.runOnce(cfg);

                    double coordElapsed = r.getCoordinator() != null ? r.getCoordinator().getElapsedSec() : Double.NaN;

                    w.write(String.format(
                            "%d,%.3f,%.3f,%.3f,%.6f,%.6f,%d,%d,%f\n",
                            cfg.N_WORKERS,
                            r.getTotalElapsedSec(),
                            coordElapsed,
                            r.lastWorkerElapsedSec(),
                            r.getCoordinator() != null ? r.getCoordinator().getGlobalBestAcc() : Double.NaN,
                            r.getCoordinator() != null ? r.getCoordinator().getGlobalBestLoss() : Double.NaN,
                            r.maxMessagesSent(),
                            r.maxBytesSent(),
                            theshold_offset
                    ));

                    w.flush();
                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with theshold_diff: " + theshold_offset);
                    System.out.println("===============================================================================================");

                }   
            }
        }
    }

    // =============================================================================================================

    private static Path createUniqueCsvPath(String baseDirName, String baseFileName) throws IOException {

        Path dir = Path.of(baseDirName);
        Files.createDirectories(dir);

        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        int rnd = ThreadLocalRandom.current().nextInt(1000, 10000); // 4 digits

        Path csv = dir.resolve(baseFileName + "_" + ts + "_" + rnd + ".csv");

        int attempt = 0;
        while (Files.exists(csv)) {
            rnd = ThreadLocalRandom.current().nextInt(1000, 10000);
            csv = dir.resolve(baseFileName + "_" + ts + "_" + rnd + ".csv");
            
            if (++attempt > 50) throw new IOException("Could not create unique CSV under " + baseDirName);
        }

        return csv;
    }

    // =============================================================================================================

    private static Path createUniqueResultsFiles(String baseDirName) throws IOException {

        Files.createDirectories(Path.of(baseDirName));

        String ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
        int rnd = ThreadLocalRandom.current().nextInt(1000, 10000); // 4 digits

        Path runDir = Path.of(baseDirName, "run_" + ts + "_" + rnd);

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
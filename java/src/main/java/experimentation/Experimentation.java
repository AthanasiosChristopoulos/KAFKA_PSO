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
    private static String bootstrap = cfg.KAFKA_HOST;
    private static float LOSS_THRESHOLD_MIN_ORIGINAL;
    private static float LOSS_THRESHOLD_MAX_ORIGINAL;
    // private static List<Integer> workersList = List.of(2, 4, 6);
    // private static List<Integer> workersList = List.of(2, 6, 12);
    // private static List<Integer> workersList = List.of(2, 12, 24); // make sure that INDEPENDENT_WORKER_DATA_PROCESSING == false
    // private static List<Integer> workersList = List.of(1, 2, 6, 12); // ignore 1 (warm up) just see 2, 12, 24
    // private static List<Integer> workersList = List.of(1, 2, 6, 12, 16); // ignore 1 (warm up) just see 2, 12, 24
    // private static List<Integer> workersList = List.of(1, 2, 6, 12, 16, 20); 

    // ========================================================================
    // Scenario with high workers:
    private static List<Integer> workersList = List.of(6, 18, 24);

    // ========================================================================
    private static List<Integer> filterEnableList = List.of(1, 0);

    // ========================================================================
    // private static List<Float> theshold_offset_list = List.of(0.00f, 0.1f, 0.2f);
    private static List<Float> theshold_offset_list = List.of(0.00f, 0.05f, 0.1f, 0.15f);

    // ========================================================================

    public static String header_1 = "FILTER_ENABLED,N_WORKERS,TOTAL_ELAPSED,COORD_ELAPSED,LAST_WORKER_ELAPSED," +
                    "GBEST_ACC,GBEST_LOSS," + 
                    "TOTAL_MESSAGES_SENT,TOTAL_MESSAGES_SENT_PBEST,TOTAL_MESSAGES_SENT_CURRENT_WEIGHTS," + 
                    "TOTAL_BYTES_SENT,LOSS_THRESHOLD_DIFF,LOSS_THRESHOLD_MIN,LOSS_THRESHOLD_MAX\n";

    public static void main(String[] args) throws Exception {

        cfg.LOSS_THRESHOLD_MAX = 0.05f;
        cfg.LOSS_THRESHOLD_MIN = 0.005f;
        LOSS_THRESHOLD_MIN_ORIGINAL = cfg.LOSS_THRESHOLD_MIN;
        LOSS_THRESHOLD_MAX_ORIGINAL = cfg.LOSS_THRESHOLD_MAX;

        if(cfg.EXPERIMENTATION_MODE.equals("N_WORKERS")) {
            
            // Path csvPath = createUniqueCsvPath(cfg.EXPERIMENTATION_DIR, "results");
            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_n_workers.csv");

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write(header_1);

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
                        topics = List.of(cfg.GPEST_WEIGHTS_TOPIC);
                    }

                    KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                    // =================================================================================================

                    ExperimentResult r = SimulationRunner.runOnce(cfg);

                    writeExperimentData(w, r, n, -1, -1);

                    w.flush();
                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with N_WORKERS: " + n);
                    System.out.println("===============================================================================================");

                }
            } 
        
        // ===========================================================================================================================================
        // ===========================================================================================================================================

        } else if(cfg.EXPERIMENTATION_MODE.equals("FILTER_ENABLED")) {
            
            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_filter_enabled.csv");

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write(header_1);

                for (int fE : filterEnableList) {

                    cfg.refreshRunId();

                    if(fE == 0) {
                        cfg.FILTER_ENABLED = false;

                    } else {
                        cfg.FILTER_ENABLED = true;
                    }

                    cfg.refreshFilterEnabled();

                    CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);

                    System.out.println("===============================================================================================");
                    System.out.println("FILTER_ENABLED: " + cfg.FILTER_ENABLED);
                    System.out.println("New RUN_ID: " + cfg.RUN_ID);
                    System.out.println("===============================================================================================");

                    // =================================================================================================
                    // Restart the Kafka Parititions

                    List<String> topics;
                    if (cfg.FULLY_INFORMED || cfg.ENABLE_NEIGHBORHOODS) {
                        topics = List.of(cfg.PBEST_WEIGHTS_TOPIC);
                    } else {
                        topics = List.of(cfg.GPEST_WEIGHTS_TOPIC);
                    }

                    KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                    // =================================================================================================

                    ExperimentResult r = SimulationRunner.runOnce(cfg);

                    writeExperimentData(w, r, -1, fE, -1);

                    w.flush();
                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with FILTER_ENABLED: " + cfg.FILTER_ENABLED);
                    System.out.println("===============================================================================================");

                }
            } 
        
        // ===========================================================================================================================================
        // ===========================================================================================================================================

        } else if(cfg.EXPERIMENTATION_MODE.equals("THRESHOLD")){
                         
            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_threshold.csv");

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write(header_1);

                for (float theshold_offset : theshold_offset_list) {

                    cfg.refreshRunId();
                    cfg.LOSS_THRESHOLD_MIN = LOSS_THRESHOLD_MIN_ORIGINAL + theshold_offset;
                    cfg.LOSS_THRESHOLD_MAX = LOSS_THRESHOLD_MAX_ORIGINAL + theshold_offset;
                    cfg.PBEST_DEBOUNCE_MS = 0;

                    CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);

                    System.out.println("===============================================================================================");
                    System.out.println("LOSS_THRESHOLD_MIN = " + cfg.LOSS_THRESHOLD_MIN + 
                        ", LOSS_THRESHOLD_MAX = " + cfg.LOSS_THRESHOLD_MAX + ", DIFF = " + theshold_offset);
                    System.out.println("New RUN_ID: " + cfg.RUN_ID);
                    System.out.println("===============================================================================================");

                    // =================================================================================================
                    // Restart the Kafka Parititions

                    List<String> topics;
                    if (cfg.FULLY_INFORMED || cfg.ENABLE_NEIGHBORHOODS) {
                        topics = List.of(cfg.PBEST_WEIGHTS_TOPIC);
                    } else {
                        topics = List.of(cfg.GPEST_WEIGHTS_TOPIC);
                    }

                    KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                    // =================================================================================================

                    ExperimentResult r = SimulationRunner.runOnce(cfg);

                    writeExperimentData(w, r, -1, -1, theshold_offset);

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

    private static void writeExperimentData(BufferedWriter w, ExperimentResult r, int nWorkers_arg, 
        int filterEnabled_arg, float theshold_offset_arg) {

        int nWorkers = cfg.N_WORKERS;
        int filterEnabled = (cfg.FILTER_ENABLED ? 1 : 0);
        float theshold_offset = cfg.LOSS_THRESHOLD_MAX - cfg.LOSS_THRESHOLD_MIN;

        if(nWorkers_arg != -1) nWorkers = nWorkers_arg;
        if(filterEnabled_arg != -1) filterEnabled = filterEnabled_arg;
        if(theshold_offset_arg != -1) theshold_offset = theshold_offset_arg;
        
        try{     

            w.write(String.format(
                "%d,%d,%.3f,%.3f,%.3f,%.6f,%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f\n",
                filterEnabled,
                nWorkers,
                r.getTotalElapsedSec(),
                r.getCoordinator() != null ? r.getCoordinator().getElapsedSec() : Double.NaN,
                r.lastWorkerElapsedSec(),
                r.getCoordinator() != null ? r.getCoordinator().getGlobalBestAcc() : Double.NaN,
                r.getCoordinator() != null ? r.getCoordinator().getGlobalBestLoss() : Double.NaN,
                r.sumMessagesSent(),
                r.sumPBestMessagesSent(),
                r.sumCurrentWeightsMessagesSent(),
                r.sumBytesSent(),
                theshold_offset,
                (double) cfg.LOSS_THRESHOLD_MIN,
                (double) cfg.LOSS_THRESHOLD_MAX
            ));

        } catch(Exception e) {
            e.printStackTrace();
        }


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
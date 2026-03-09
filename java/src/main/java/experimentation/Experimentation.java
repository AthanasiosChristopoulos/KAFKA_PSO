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

    private static volatile boolean experimentationStopRequested = false;
    private static volatile boolean ctrlCRequested = false;

    private static final Config cfg = Config.getInstance();
    private static String bootstrap = cfg.KAFKA_HOST;
    private static float LOSS_THRESHOLD_MIN_ORIGINAL;
    private static float LOSS_THRESHOLD_MAX_ORIGINAL;

    // ========================================================================

    public static String header_1 = "`,N_WORKERS,TOTAL_ELAPSED,COORD_ELAPSED,LAST_WORKER_ELAPSED," +
                    "GBEST_ACC,GBEST_LOSS," + 
                    "TOTAL_MESSAGES_SENT,TOTAL_MESSAGES_SENT_PBEST,TOTAL_MESSAGES_SENT_CURRENT_WEIGHTS," + 
                    "TOTAL_BYTES_SENT,LOSS_THRESHOLD_DIFF,LOSS_THRESHOLD_MIN,LOSS_THRESHOLD_MAX," +
                    "PBEST_DEBOUNCE_MS,MONITORING_THRESHOLD_MIN,MONITORING_THRESHOLD_MAX\n";

    // ========================================================================

    private static void installSigintHandler() {
        try {
            // HotSpot: available on most JDKs (including Windows)
            sun.misc.Signal.handle(new sun.misc.Signal("INT"), sig -> {
                if (ctrlCRequested) return;
                ctrlCRequested = true;

                System.out.println("\n[Experimentation] Ctrl+C caught -> stopping current run (not exiting JVM)");
                // Ask your system to stop (workers/coordinator are already polling this)
                CoordinatorControl.getInstance().requestStopFinal();

                experimentationStopRequested = true;
            });
        } catch (Throwable t) {
            // Fallback: if Signal not available, you can't prevent JVM exit on Ctrl+C.
            System.out.println("[Experimentation] WARNING: sun.misc.Signal not available; Ctrl+C will terminate JVM.");
        }
    }

    // ========================================================================

    public static void main(String[] args) throws Exception {
        installSigintHandler();

        cfg.LOSS_THRESHOLD_MAX = 0.05f;
        cfg.LOSS_THRESHOLD_MIN = 0.005f;
        LOSS_THRESHOLD_MIN_ORIGINAL = cfg.LOSS_THRESHOLD_MIN;
        LOSS_THRESHOLD_MAX_ORIGINAL = cfg.LOSS_THRESHOLD_MAX;
        
        // ===========================================================================================================================================
        // ===========================================================================================================================================

        if(cfg.EXPERIMENTATION_MODE.equals("N_WORKERS")) {
            
            // Path csvPath = createUniqueCsvPath(cfg.EXPERIMENTATION_DIR, "results");
            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_n_workers.csv");
            // List<Integer> workersList = List.of(2, 4, 6);
            // List<Integer> workersList = List.of(2, 6, 12);
            // List<Integer> workersList = List.of(2, 12, 24); // make sure that INDEPENDENT_DATA_PROCESSING == false
            // List<Integer> workersList = List.of(1, 2, 6, 12); // ignore 1 (warm up) just see 2, 12, 24
            // List<Integer> workersList = List.of(1, 2, 6, 12, 16); // ignore 1 (warm up) just see 2, 12, 24
            List<Integer> workersList = List.of(1, 2, 6, 12, 16, 20); 
            // List<Integer> workersList = List.of(1, 16);
            // ========================================================================
            // Scenario with high workers:
            
            // List<Integer> workersList = List.of(6, 18, 24);

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write(header_1);

                for (int n : workersList) {
                    
                    cfg.refreshRunId();
                    CustomLogger.refreshAll();
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
                        topics = List.of(cfg.PBEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    } else {
                        topics = List.of(cfg.GPEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    }

                    KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                    // =================================================================================================

                    ExperimentResult r = SimulationRunner.runOnce(cfg);

                    writeExperimentData(w, r, n, -1, -1);

                    w.flush();
                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with N_WORKERS: " + n);
                    System.out.println("===============================================================================================");

                    if(experimentationStopRequested == true) {
                        System.exit(0);
                    }
                }
            } 
        
        // ===========================================================================================================================================
        // ===========================================================================================================================================

        } else if(cfg.EXPERIMENTATION_MODE.equals("FILTER_ENABLED")) {
            
            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_filter_enabled.csv");

            // ========================================================================
            // List<Integer> filterEnableList = List.of(1, 0);
            List<Integer> filterEnableList = List.of(0, 1);
            // List<Integer> filterEnableList = List.of(1);
            
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
                    CustomLogger.refreshAll();
                    CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);

                    System.out.println("===============================================================================================");
                    System.out.println("FILTER_ENABLED: " + cfg.FILTER_ENABLED);
                    System.out.println("New RUN_ID: " + cfg.RUN_ID);
                    System.out.println("===============================================================================================");

                    // =================================================================================================
                    // Restart the Kafka Parititions

                    List<String> topics;
                    if (cfg.FULLY_INFORMED || cfg.ENABLE_NEIGHBORHOODS) {
                        topics = List.of(cfg.PBEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    } else {
                        topics = List.of(cfg.GPEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    }

                    KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                    // =================================================================================================

                    ExperimentResult r = SimulationRunner.runOnce(cfg);

                    writeExperimentData(w, r, -1, fE, -1);

                    w.flush();
                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with FILTER_ENABLED: " + cfg.FILTER_ENABLED);
                    System.out.println("===============================================================================================");
                    if(experimentationStopRequested == true) {
                        System.exit(0);
                    }
                }
            } 
        
        // ===========================================================================================================================================
        // ===========================================================================================================================================

        } else if(cfg.EXPERIMENTATION_MODE.equals("THRESHOLD")){
                         
            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_threshold.csv");

            // ========================================================================
            // List<Float> theshold_offset_list = List.of(0.00f, 0.1f, 0.2f);
            List<Float> theshold_offset_list = List.of(0.00f, 0.05f, 0.1f, 0.15f);

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write(header_1);

                for (float theshold_offset : theshold_offset_list) {

                    cfg.FILTER_ENABLED = true;
                    cfg.refreshRunId();
                    CustomLogger.refreshAll();
                    cfg.refreshFilterEnabled();

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
                        topics = List.of(cfg.PBEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    } else {
                        topics = List.of(cfg.GPEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    }

                    KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                    // =================================================================================================

                    ExperimentResult r = SimulationRunner.runOnce(cfg);

                    writeExperimentData(w, r, -1, -1, theshold_offset);

                    w.flush();
                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with theshold_diff: " + theshold_offset);
                    System.out.println("===============================================================================================");
                    if(experimentationStopRequested == true) {
                        System.exit(0);
                    }
                }   
            }

        // ===========================================================================================================================================
        // ===========================================================================================================================================

        } else if(cfg.EXPERIMENTATION_MODE.equals("MONITORING_ITERATIONS")) {
            
            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_monitoring_iterations.csv");

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write("MONITORING_ITER,TIME_SEC,ACCURACY\n");

                CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);
                CustomLogger.refreshAll();

                System.out.println("===============================================================================================");
                System.out.println("New RUN_ID: " + cfg.RUN_ID);
                System.out.println("===============================================================================================");

                // =================================================================================================
                // Restart the Kafka Parititions

                List<String> topics;
                if (cfg.FULLY_INFORMED || cfg.ENABLE_NEIGHBORHOODS) {
                    topics = List.of(cfg.PBEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                } else {
                    topics = List.of(cfg.GPEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                }

                KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                // =================================================================================================
                ExperimentResult r = null;

                r = SimulationRunner.runOnce(cfg);

                // writeExperimentData(w, r, -1, -1, -1);
                writeAccuracyValues(w, r);

                w.flush();

                System.out.println("===============================================================================================");
                System.out.println("End of experiment with MONITORING_ITERATIONS");
                System.out.println("===============================================================================================");
                if(experimentationStopRequested == true) {
                    System.exit(0);
                }
            }

        // =========================================================================================================================================
        // =========================================================================================================================================
        // =========================================================================================================================================

        } else if (cfg.EXPERIMENTATION_MODE.equals("SEVERITY_OF_FILTER")) {

            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_severity_of_filter.csv");

            var severities = List.of(
                FilterSeverity.Level.OFF,
                FilterSeverity.Level.EASY,
                FilterSeverity.Level.MEDIUM,
                FilterSeverity.Level.HARD
            );
            
            // var severities = List.of(
            //     FilterSeverity.Level.OFF,
            //     FilterSeverity.Level.EASY
            // );

            // var severities = List.of(
            //     FilterSeverity.Level.OFF
            // );

            // var severities = List.of(
            //     FilterSeverity.Level.EASY
            // );

            String header = "SEVERITY_CODE,SEVERITY_NAME," + header_1; 

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write(header);

                for (FilterSeverity.Level level : severities) {

                    cfg.refreshRunId();
                    CustomLogger.refreshAll();

                    FilterSeverity.apply(cfg, level);

                    CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);

                    System.out.println("===============================================================================================");
                    System.out.println("SEVERITY: " + level + " (code=" + level.code + ")");
                    System.out.println("FILTER_ENABLED: " + cfg.FILTER_ENABLED);
                    System.out.println("LOSS_THRESHOLD_MIN=" + cfg.LOSS_THRESHOLD_MIN + ", LOSS_THRESHOLD_MAX=" + cfg.LOSS_THRESHOLD_MAX);
                    System.out.println("PBEST_DEBOUNCE_MS=" + cfg.PBEST_DEBOUNCE_MS);
                    System.out.println("MONITORING_THRESHOLD_MIN=" + cfg.MONITORING_THRESHOLD_MIN +
                                    ", MONITORING_THRESHOLD_MAX=" + cfg.MONITORING_THRESHOLD_MAX);
                    System.out.println("RUN_ID: " + cfg.RUN_ID);
                    System.out.println("===============================================================================================");

                    // Restart Kafka topic(s) like you already do
                    List<String> topics;
                    if (cfg.FULLY_INFORMED || cfg.ENABLE_NEIGHBORHOODS) {
                        topics = List.of(cfg.PBEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    } else {
                        topics = List.of(cfg.GPEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    }
                    KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                    ExperimentResult r = SimulationRunner.runOnce(cfg);

                    // Write severity info + existing metrics
                    w.write(String.format("%d,%s,", level.code, level.name()));
                    writeExperimentData(w, r, -1, (cfg.FILTER_ENABLED ? 1 : 0), -1);

                    w.flush();

                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with SEVERITY: " + level);
                    System.out.println("===============================================================================================");

                    if(experimentationStopRequested == true) {
                        System.exit(0);
                    }
                }
            }

        // ================================================================================================

        } else if (cfg.EXPERIMENTATION_MODE.equals("FULLY_INFORMED_VS_CLASSICAL")) {

            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_fully_informed_vs_classical.csv");

            List<Boolean> fully_informed_list = List.of(false, true);
        
            String header = "FULLY_INFORMED," + header_1; 

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write(header);

                for (Boolean fully_informed : fully_informed_list) {

                    cfg.refreshRunId();
                    CustomLogger.refreshAll();
                    CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);

                    cfg.FULLY_INFORMED = fully_informed;

                    System.out.println("===============================================================================================");
                    System.out.println("FULLY_INFORMED: " + cfg.FULLY_INFORMED);
                    System.out.println("RUN_ID: " + cfg.RUN_ID);
                    System.out.println("===============================================================================================");

                    // Restart Kafka topic(s) like you already do
                    List<String> topics;
                    if (cfg.FULLY_INFORMED || cfg.ENABLE_NEIGHBORHOODS) {
                        topics = List.of(cfg.PBEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    } else {
                        topics = List.of(cfg.GPEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    }
                    KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                    ExperimentResult r = SimulationRunner.runOnce(cfg);

                    // Write severity info + existing metrics
                    w.write(String.format("%b,", cfg.FULLY_INFORMED));
                    writeExperimentData(w, r, -1, -1, -1);

                    w.flush();

                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with cfg.FULLY_INFORMED: " + cfg.FULLY_INFORMED);
                    System.out.println("===============================================================================================");

                    if(experimentationStopRequested == true) {
                        System.exit(0);
                    }
                }
            }
        }
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
                "%d,%d,%.3f,%.3f,%.3f,%.6f,%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%d,%d\n",
                filterEnabled,
                nWorkers,
                // r.getTotalElapsedSec(),
                r.lastWorkerElapsedSec(),        
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
                (double) cfg.LOSS_THRESHOLD_MAX,
                cfg.PBEST_DEBOUNCE_MS,
                cfg.MONITORING_THRESHOLD_MIN,
                cfg.MONITORING_THRESHOLD_MAX

            ));

        } catch(Exception e) {
            e.printStackTrace();
        }
    }


    // =============================================================================================================

    private static void writeAccuracyValues(BufferedWriter w, ExperimentResult r) {
        try {
            int iter = 0;

            for (AccuracyPoint p : r.getAccuracyValues()) {
                iter++;
                w.write(String.format(
                    "%d,%f,%f\n",
                    iter,
                    p.getElapsedSec(),
                    p.getAccuracy()
                ));
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
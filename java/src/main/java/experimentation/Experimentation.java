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

    public static String header_1 = "FILTER_ENABLED,N_WORKERS,TOTAL_ELAPSED,COORD_ELAPSED,LAST_WORKER_ELAPSED," +
                    "GBEST_ACC,GBEST_LOSS," + 
                    "TOTAL_MESSAGES_SENT,TOTAL_MESSAGES_SENT_PBEST,TOTAL_MESSAGES_SENT_CURRENT_WEIGHTS," + 
                    "TOTAL_BYTES_SENT,LOSS_THRESHOLD_DIFF,LOSS_THRESHOLD_MIN,LOSS_THRESHOLD_MAX," +
                    "PBEST_DEBOUNCE_MS,MONITORING_THRESHOLD_MIN,MONITORING_THRESHOLD_MAX," +
                    "INDEPENDENT_DATA_PROCESSING,ENABLE_NEIGHBORHOODS\n";

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

        cfg.EARLY_STOPPING = false; // this is detrimental to experimentation, especially in the time diagrams:
            // N_WORKERS => on low N_WORKERS, workers exit sooner than expected, because they have pretty bad performances
                // EARLY STOPPING kicks in
            // STRENGTH => Easier filters reach EARLY_STOPPING requiring less time, when Harder filters require more 
                // time to reach good accuracy but they are contantly improving but slowly, so no EARLY_STOPPING
    
        // ========================================================
        // =====================================================

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
            
            // List<Integer> workersList = List.of(6, 12, 18, 24, 30);

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write(header_1);

                for (int n : workersList) {
                    
                    cfg.refreshConfig();
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
                    if (cfg.FULLY_INFORMED || (cfg.ENABLE_NEIGHBORHOODS)) {
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
        
        // ===============================================================
        // ============================================================

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

                    if(fE == 0) {
                        cfg.FILTER_ENABLED = false;

                    } else {
                        cfg.FILTER_ENABLED = true;
                    }

                    cfg.refreshConfig();
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
        
        // ===============================================================
        // ==============================================================

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
                    CustomLogger.refreshAll();
                    cfg.refreshConfig();

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

        // ========================================================================
        // ====================================================================

        } else if(cfg.EXPERIMENTATION_MODE.equals("MONITORING_ITERATIONS")) {

            cfg.EARLY_STOPPING = true; 
            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_monitoring_iterations.csv");

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write("MONITORING_ITER,TIME_SEC,ACCURACY\n");  // this is independent, this is the last one, no more columns

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
                w.write(header_1);
                writeExperimentData(w, r, -1, -1, -1);

                w.flush();
                
                System.out.println("===============================================================================================");
                System.out.println("End of experiment with MONITORING_ITERATIONS");
                System.out.println("===============================================================================================");
                if(experimentationStopRequested == true) {
                    System.exit(0);
                }
            }

        // ==================================================================================================
        // ====================================================================================================
        // =======================================================================================================

        } else if(cfg.EXPERIMENTATION_MODE.equals("LOSS_FUNCTIONS")) {

            cfg.EARLY_STOPPING = true; 
            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_loss_functions.csv");

            // List<String> loss_function_list = List.of("HINGE", "MAE", "CROSS_ENTROPY", "CROSS_ENTROPY");
            // List<String> combine_loss_list = List.of("AVG", "TOPK", "AVG", "AVG");
            // List<String> regularizer_list = List.of("NONE", "NONE", "SLOPE", "NONE");

            // List<String> loss_function_list = List.of("CROSS_ENTROPY");
            // List<String> combine_loss_list = List.of("AVG");
            // List<String> regularizer_list = List.of("NONE");

            // List<String> loss_function_list = List.of("CROSS_ENTROPY", "CROSS_ENTROPY");
            // List<String> combine_loss_list = List.of("AVG", "AVG");
            // List<String> regularizer_list = List.of("SLOPE", "NONE");

            // List<String> loss_function_list = List.of("HINGE", "MAE", "CROSS_ENTROPY");
            // List<String> combine_loss_list = List.of("AVG", "TOPK", "AVG");
            // List<String> regularizer_list = List.of("NONE", "NONE", "NONE");
            
            // List<String> loss_function_list = List.of("CROSS_ENTROPY", "MAE");
            // List<String> combine_loss_list = List.of("AVG", "TOPK");
            // List<String> regularizer_list = List.of("NONE", "NONE");

            // List<String> loss_function_list = List.of("CROSS_ENTROPY", "CROSS_ENTROPY");
            // List<String> combine_loss_list = List.of("AVG", "AVG");
            // List<String> regularizer_list = List.of("NONE", "NONE");

            // List<String> loss_function_list = List.of("MAE", "CROSS_ENTROPY");
            // List<String> combine_loss_list = List.of("AVG", "AVG");
            // List<String> regularizer_list = List.of("NONE", "NONE");

            // List<String> loss_function_list = List.of("HINGE", "CROSS_ENTROPY");
            // List<String> combine_loss_list = List.of("AVG", "AVG");
            // List<String> regularizer_list = List.of("NONE", "NONE");

            // List<String> loss_function_list = List.of("HINGE", "MAE", "CROSS_ENTROPY", "CROSS_ENTROPY");
            // List<String> combine_loss_list = List.of("AVG", "TOPK", "AVG", "AVG");
            // List<String> regularizer_list = List.of("NONE", "NONE", "SLOPE", "NONE");

            // List<String> loss_function_list = List.of("CROSS_ENTROPY", "HINGE", "MAE");
            // List<String> combine_loss_list = List.of("AVG", "AVG", "TOPK");
            // List<String> regularizer_list = List.of("SLOPE", "NONE", "NONE");

            // List<String> loss_function_list = List.of("CROSS_ENTROPY", "MAE");
            // List<String> combine_loss_list = List.of("AVG", "TOPK");
            // List<String> regularizer_list = List.of("NONE", "NONE");

            List<String> loss_function_list = List.of("MAE");
            List<String> combine_loss_list = List.of("TOPK");
            List<String> regularizer_list = List.of("NONE");

            // List<String> loss_function_list = List.of("CROSS_ENTROPY", "HINGE");
            // List<String> combine_loss_list = List.of("AVG", "AVG");
            // List<String> regularizer_list = List.of("NONE", "NONE");

            // List<String> loss_function_list = List.of("HINGE");
            // List<String> combine_loss_list = List.of("AVG");
            // List<String> regularizer_list = List.of("NONE");

            ExperimentResult r = null;
            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                for (int i = 0; i < loss_function_list.size(); i++) {
                    String loss_function = loss_function_list.get(i);
                    String combine_loss = combine_loss_list.get(i);
                    String regularizer = regularizer_list.get(i);

                    w.write("MONITORING_ITER,TIME_SEC,ACCURACY\n");  // this is independent, this is the last one, no more columns

                    cfg.LOSS_FUNCTION = loss_function;
                    cfg.COMBINE_LOSS = combine_loss;
                    cfg.REGULARIZER = regularizer;

                    CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);
                    CustomLogger.refreshAll();
                    cfg.refreshConfig();

                    System.out.println("==================================================================================");
                    System.out.println("New RUN_ID: " + cfg.RUN_ID);
                    System.out.println("==================================================================================");

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

                    r = SimulationRunner.runOnce(cfg);
                    writeAccuracyValues(w, r);
                    w.write(String.format("%s,%s,%s,%s,%s,%s\n",
                        "LOSS_FUNCTION", cfg.LOSS_FUNCTION,
                        "COMBINE_LOSS", cfg.COMBINE_LOSS,
                        "REGULARIZER", cfg.REGULARIZER));
                    w.flush();
                    r.resetAccuracyValues();

                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with LOSS_FUNCTION: " + cfg.LOSS_FUNCTION + 
                        ", and COMBINE_LOSS: " + cfg.COMBINE_LOSS + ", and REGULARIZER: " + cfg.REGULARIZER);
                    System.out.println("===============================================================================================");
                    if(experimentationStopRequested == true) {
                        System.exit(0);
                    }
                }

                w.write(header_1);
                writeExperimentData(w, r, -1, -1, -1);
            }

        // ==================================================================================================
        // ====================================================================================================
        // =======================================================================================================

        } else if (cfg.EXPERIMENTATION_MODE.equals("FILTER_STRENGTH")) {

            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_strength.csv");

            // =====================================================================
            var severities = List.of(
                FilterStrength.Level.OFF,
                FilterStrength.Level.EASY,
                FilterStrength.Level.MEDIUM,
                FilterStrength.Level.HARD
            );
            
            // var severities = List.of(
            //     FilterStrength.Level.OFF,
            //     FilterStrength.Level.EASY
            // );

            // var severities = List.of(
            //     FilterStrength.Level.OFF
            // );

            // var severities = List.of(
            //     FilterStrength.Level.EASY
            // );
            // =====================================================================

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write("STRENGTH_CODE,STRENGTH_NAME," + header_1);

                for (FilterStrength.Level level : severities) {

                    cfg.refreshConfig();
                    CustomLogger.refreshAll();
                    FilterStrength.apply(cfg, level);
                    CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);

                    System.out.println("===============================================================================");
                    System.out.println("STRENGTH: " + level + " (code=" + level.code + ")");
                    System.out.println("FILTER_ENABLED: " + cfg.FILTER_ENABLED);
                    System.out.println("LOSS_THRESHOLD_MIN=" + cfg.LOSS_THRESHOLD_MIN + ", LOSS_THRESHOLD_MAX=" + cfg.LOSS_THRESHOLD_MAX);
                    System.out.println("PBEST_DEBOUNCE_MS=" + cfg.PBEST_DEBOUNCE_MS);
                    System.out.println("MONITORING_THRESHOLD_MIN=" + cfg.MONITORING_THRESHOLD_MIN +
                                    ", MONITORING_THRESHOLD_MAX=" + cfg.MONITORING_THRESHOLD_MAX);
                    System.out.println("RUN_ID: " + cfg.RUN_ID);
                    System.out.println("===============================================================================");

                    // Restart Kafka topic(s) like you already do
                    List<String> topics;
                    if (cfg.FULLY_INFORMED || cfg.ENABLE_NEIGHBORHOODS) {
                        topics = List.of(cfg.PBEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    } else {
                        topics = List.of(cfg.GPEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
                    }
                    KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

                    ExperimentResult r = SimulationRunner.runOnce(cfg);

                    // Write strength info + existing metrics
                    w.write(String.format("%d,%s,", level.code, level.name()));
                    writeExperimentData(w, r, -1, (cfg.FILTER_ENABLED ? 1 : 0), -1);

                    w.flush();

                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with STRENGTH: " + level);
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

            // =====================================================================
            List<Boolean> fully_informed_list = List.of(false, true);
            // =====================================================================

            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write("FULLY_INFORMED," + header_1);

                for (Boolean fully_informed : fully_informed_list) {

                    CustomLogger.refreshAll();
                    CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);

                    cfg.FULLY_INFORMED = fully_informed;
                    cfg.ENABLE_NEIGHBORHOODS = false;
                    cfg.N_WORKERS = 10;
                    cfg.refreshConfig();

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

        // ================================================================================================

        } else if (cfg.EXPERIMENTATION_MODE.equals("DIMENSIONALITY")) {

            // cfg.EARLY_STOPPING = true;
            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_dimensionality.csv");

            // =====================================================================
            // List<Integer> model_version_list = List.of(1, 2, 3, 4);
            List<Integer> model_version_list = List.of(1, 4);

            // =====================================================================
            
            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write("MODEL_VERSION,DIMENSIONALITY," + header_1);

                for (Integer model_version : model_version_list) {

                    CustomLogger.refreshAll();
                    CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);

                    cfg.MODEL_VERSION = model_version;
                    cfg.refreshConfig();

                    System.out.println("===============================================================================================");
                    System.out.println("MODEL_VERSION: " + cfg.MODEL_VERSION);
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

                    w.write(String.format("%d,%d,", cfg.MODEL_VERSION, r.getDimensionality()));
                    writeExperimentData(w, r, -1, -1, -1);

                    w.flush();

                    System.out.println("===============================================================================================");
                    System.out.println("End of experiment with cfg.MODEL_VERSION: " + cfg.MODEL_VERSION +
                        ", with cfg.DIMENSIONALITY: " + r.getDimensionality()
                    );
                    System.out.println("===============================================================================================");

                    if(experimentationStopRequested == true) {
                        System.exit(0);
                    }
                }
            } 

        // ================================================================================================

        } else if (cfg.EXPERIMENTATION_MODE.equals("TOPOLOGY")) {

            cfg.EARLY_STOPPING = true;
            Path dir = Path.of(cfg.EXPERIMENTATION_DIR);
            Files.createDirectories(dir);
            Path csvPath = dir.resolve("results_topology.csv");

            // =====================================================================
            List<String> topology_list = List.of("all", "ring", "square");
            // List<String> topology_list = List.of("all");

            // =====================================================================
            
            try (BufferedWriter w = Files.newBufferedWriter(
                    csvPath,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            )) {
                w.write("ENABLE_NEIGHBORHOODS,NEIGHBORHOOD_TOPOLOGY," + header_1);

                for (String topology : topology_list) {

                    CustomLogger.refreshAll();
                    CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);

                    cfg.ENABLE_NEIGHBORHOODS = true;
                    cfg.NEIGHBORHOOD_TOPOLOGY = topology;
                    cfg.refreshConfig();

                    System.out.println("===============================================================================================");
                    System.out.println("ENABLE_NEIGHBORHOODS: " + cfg.ENABLE_NEIGHBORHOODS + 
                        ", NEIGHBORHOOD_TOPOLOGY: " + cfg.NEIGHBORHOOD_TOPOLOGY);
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

                    w.write(String.format("%s,%s,", cfg.ENABLE_NEIGHBORHOODS, cfg.NEIGHBORHOOD_TOPOLOGY));
                    writeExperimentData(w, r, -1, -1, -1);

                    w.flush();

                    System.out.println("===============================================================================================");
                    System.out.println("End of Experiment with ENABLE_NEIGHBORHOODS: " + cfg.ENABLE_NEIGHBORHOODS + 
                        ", NEIGHBORHOOD_TOPOLOGY: " + cfg.NEIGHBORHOOD_TOPOLOGY);
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
                "%d,%d,%.3f,%.3f,%.3f,%.6f,%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%d,%d,%b,%b\n",
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
                cfg.MONITORING_THRESHOLD_MAX,
                cfg.INDEPENDENT_DATA_PROCESSING,
                cfg.ENABLE_NEIGHBORHOODS
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
                    "%d,%f,%f\n",   // this is independent, this is the last one, no more columns
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
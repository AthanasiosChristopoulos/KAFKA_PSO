package experimentation;
import utils.*;
import state.*;

import java.io.BufferedWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.nio.file.*;

public class ExperimentationCoordinator {

    private static volatile boolean experimentationStopRequested = false;
    private static volatile boolean ctrlCRequested = false;

    private static final Config cfg = Config.getInstance();
    private static String bootstrap = cfg.KAFKA_HOST;

    // ========================================================================

    public static String header_1 = "FILTER_ENABLED,N_WORKERS,TOTAL_ELAPSED,COORD_ELAPSED,LAST_WORKER_ELAPSED," +
                    "GBEST_ACC,GBEST_LOSS," + 
                    "TOTAL_MESSAGES_SENT,TOTAL_MESSAGES_SENT_PBEST,TOTAL_MESSAGES_SENT_CURRENT_WEIGHTS," + 
                    "TOTAL_BYTES_SENT,LOSS_THRESHOLD_DIFF,LOSS_THRESHOLD_MIN,LOSS_THRESHOLD_MAX," +
                    "PBEST_DEBOUNCE_MS,MONITORING_THRESHOLD_MIN,MONITORING_THRESHOLD_MAX," +
                    "DATASET_PARTITIONING,EARLY_STOPPING,HEAVY_SAMPLES,DATASET,ENABLE_NEIGHBORHOODS,TOPOLOGY,ND4J_PROFILE,AVG_NETWORK_DELAY\n";

    // ========================================================================

    private static void installSigintHandler() {
        try {
            sun.misc.Signal.handle(new sun.misc.Signal("INT"), sig -> {
                if (ctrlCRequested) return;
                ctrlCRequested = true;

                System.out.println("\n[Experimentation] Ctrl+C caught -> stopping current run (not exiting JVM)");
                CoordinatorControl.getInstance().requestStopFinal();

                experimentationStopRequested = true;
            });

        } catch (Throwable t) {
            System.out.println("[Experimentation] WARNING: sun.misc.Signal not available; Ctrl+C will terminate JVM.");
        }
    }

    // ========================================================================

    public static void main(String[] args) throws Exception {
        installSigintHandler();
        cfg.FEDERATED_SETTING = true;

        cfg.LOSS_THRESHOLD_MAX = 0.05f;
        cfg.LOSS_THRESHOLD_MIN = 0.005f;

        cfg.EARLY_STOPPING = false; 

        if(cfg.EXPERIMENTATION_MODE.equals("MONITORING_ITERATIONS")) {

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
                w.write("MONITORING_ITER,TIME_SEC,ACCURACY\n"); 

                CoordinatorControl.getInstance().resetForNewRun(cfg.N_WORKERS);
                CustomLogger.refreshAll();

                System.out.println("=================================================");
                System.out.println("New RUN_ID: " + cfg.RUN_ID);
                System.out.println("=================================================");

                // ================================================================
                // Restart the Kafka Parititions
                resetTopics();
                // =================================================================
                printExperimentData(-1, -1, -1);

                ExperimentResult r = null;
                r = SimulationRunnerCoordinator.runOnce(cfg);

                writeAccuracyValues(w, r);
                w.write(header_1);
                writeExperimentData(w, r, -1, -1, -1);

                w.flush();
                
                System.out.println("=================================================");
                System.out.println("End of experiment with MONITORING_ITERATIONS");
                System.out.println("=================================================");

                if(experimentationStopRequested == true) {
                    System.exit(0);
                }
            }
        }
    }

    // ==============================================================================================

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
                "%d,%d,%.3f,%.3f,%.3f,%.6f,%.6f,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%d,%d,%b,%b,%b,%b,%s,%b,%s,%.6f\n",
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
                cfg.DATASET_PARTITIONING,
                cfg.EARLY_STOPPING,
                cfg.HEAVY_SAMPLES,
                cfg.DATASET,
                cfg.ENABLE_NEIGHBORHOODS,
                cfg.NEIGHBORHOOD_TOPOLOGY, 
                cfg.ND4J_PROFILE,
                r.getAvgNetworkDelay()
            ));

        } catch(Exception e) {
            e.printStackTrace();
        }
    }

    // =============================================================================================================

    private static void printExperimentData(
        int nWorkers_arg,
        int filterEnabled_arg,
        float theshold_offset_arg
    ) {
        int nWorkers = cfg.N_WORKERS;
        int filterEnabled = (cfg.FILTER_ENABLED ? 1 : 0);
        float theshold_offset = cfg.LOSS_THRESHOLD_MAX - cfg.LOSS_THRESHOLD_MIN;

        if (nWorkers_arg != -1) nWorkers = nWorkers_arg;
        if (filterEnabled_arg != -1) filterEnabled = filterEnabled_arg;
        if (theshold_offset_arg != -1) theshold_offset = theshold_offset_arg;

        System.out.println("filterEnabled: " + filterEnabled);
        System.out.println("nWorkers: " + nWorkers);

        System.out.println("threshold_offset: " + theshold_offset);
        System.out.println("LOSS_THRESHOLD_MIN: " + cfg.LOSS_THRESHOLD_MIN);
        System.out.println("LOSS_THRESHOLD_MAX: " + cfg.LOSS_THRESHOLD_MAX);

        System.out.println("PBEST_DEBOUNCE_MS: " + cfg.PBEST_DEBOUNCE_MS);
        System.out.println("MONITORING_THRESHOLD_MIN: " + cfg.MONITORING_THRESHOLD_MIN);
        System.out.println("MONITORING_THRESHOLD_MAX: " + cfg.MONITORING_THRESHOLD_MAX);

        System.out.println("DATASET_PARTITIONING: " + cfg.DATASET_PARTITIONING);
        System.out.println("EARLY_STOPPING: " + cfg.EARLY_STOPPING);
        System.out.println("HEAVY_SAMPLES: " + cfg.HEAVY_SAMPLES);

        System.out.println("DATASET: " + cfg.DATASET);
        System.out.println("ENABLE_NEIGHBORHOODS: " + cfg.ENABLE_NEIGHBORHOODS);
        System.out.println("NEIGHBORHOOD_TOPOLOGY: " + cfg.NEIGHBORHOOD_TOPOLOGY);
        System.out.println("ND4J_PROFILE: " + cfg.ND4J_PROFILE);
    }

    // =====================================================================================

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

    // =============================================================================

    public static void resetTopics() {
        List<String> topics = new ArrayList<>();

        if (cfg.ENABLE_NEIGHBORHOODS) {
            topics.add(cfg.LOCAL_WEIGHTS_TOPIC);

            List<String> workerTopics = IntStream.range(0, cfg.N_WORKERS)
                    .mapToObj(i -> "PBEST-WORKER-" + i)
                    .collect(Collectors.toList());

            topics.addAll(workerTopics);

        } else if (cfg.FULLY_INFORMED) {
            topics = List.of(cfg.PBEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
        } else {
            topics = List.of(cfg.GBEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
        }

        try {
            KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);
        } catch(Exception e) {
            e.printStackTrace();

        }
    }
}



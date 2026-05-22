package pso;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.nd4j.linalg.factory.Nd4j;
import dl4j_models.Dl4jModelFactory;
import dl4j_models.PsoModel;

import utils.*; 

public class Simulation {

    private static final Config cfg = Config.getInstance();

    private static String bootstrap = cfg.KAFKA_HOST;
    private static volatile long startNs = 0L;
    private static volatile long simulationStartMs = 0L;

    public static void main(String[] args) throws Exception {

        cfg.FEDERATED_SETTING = false;
        System.out.println("OK Running");
        simulationStartMs = java.time.Instant.now().toEpochMilli();
        System.out.println("Starting simulation at: " + simulationStartMs);

        // =================================================================================================
        // Restart the Kafka Parititions
        
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

        KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

        // System.exit(0);
        // =================================================================================================

        Thread.setDefaultUncaughtExceptionHandler((t, e) -> {
            System.err.println("DEFAULT uncaught in " + t.getName());
            e.printStackTrace();
        });

        // Test: =====================================================================

        PsoModel model = Dl4jModelFactory.createModel(-1, true).getFirst();

        System.out.println("OK loaded:");
        System.out.println(model.summary());

        // System.exit(0);

        // System.out.println("printCudaCudnnVersions");
        // printCudaCudnnVersions();
        // =============================================================================================

        var baseStateDir = java.nio.file.Path.of(cfg.KAFKA_TMP_DIR + "/");
        deleteDir(baseStateDir);
        java.nio.file.Files.createDirectories(baseStateDir);

        startNs = System.nanoTime();

        Config cfg = Config.getInstance();
        int numWorkers = cfg.N_WORKERS;

        System.out.println("============== Simulation Start ======================");
        System.out.println("Starting Coordinator ...");

        System.out.println("ND4J backend: " + Nd4j.getBackend().getClass().getName());
        System.out.println("ND4J ops: " + Nd4j.getExecutioner().getClass().getName());
        System.out.println("Data type: " + Nd4j.dataType());

        Coordinator coordinator = new Coordinator(null);
        Thread coordinatorThread = new Thread(coordinator, "coordinator");  // the coordinator starts first and then the workers
        coordinatorThread.start();

        System.out.println("Starting " + numWorkers + " workers ...");

        List<Thread> workerThreads = new ArrayList<>();
                                        
        for (int i = 0; i < numWorkers; i++) {
            Worker worker = new Worker(i, null);
            Thread workerThread = new Thread(worker, "worker-thread-" + i);
            workerThread.start();
            workerThreads.add(workerThread);
            System.out.println("Started worker thread " + i);
        }

        for (Thread t : workerThreads) {
            t.join();
        }

        double elapsedTimeSec = (System.nanoTime() - startNs) / 1_000_000_000.0; // 10^9, so this is converting to seconds    
        System.out.printf("============= Training is over, ElapsedTime: %.3f seconds =============%n", elapsedTimeSec);

        coordinatorThread.join(); // if finished every worker waits on the coordinator
        System.out.println("============== Coordinator stopped, stopping simulation ==============");

    }

    // ===================================================================================================
    
    private static void deleteDir(java.nio.file.Path path) throws java.io.IOException {
        if (!java.nio.file.Files.exists(path)) return;
        java.nio.file.Files.walk(path)
            .sorted(java.util.Comparator.reverseOrder())
            .forEach(p -> {
                try { 
                    java.nio.file.Files.delete(p); 
                } catch (Exception ignored) {

                }

            });
    }

    // ===================================================================================================
    
    public static long getSimulationStartMs() {
        return simulationStartMs;
    }
}

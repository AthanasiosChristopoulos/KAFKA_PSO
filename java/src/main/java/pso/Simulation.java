package pso;

import java.util.ArrayList;
import java.util.List;
import java.io.InputStream;
import java.util.logging.LogManager;
import org.nd4j.linalg.factory.Nd4j;

import utils.*; 

public class Simulation {

    public static void main(String[] args) throws Exception {
        
        var baseStateDir = java.nio.file.Path.of("/tmp/kstreams/");
        deleteDir(baseStateDir);
        java.nio.file.Files.createDirectories(baseStateDir);

        Config cfg = Config.getInstance();
        int numWorkers = cfg.N_WORKERS;

        System.out.println("============== Simulation Start ======================");
        System.out.println("Starting Coordinator ...");
        System.out.println("ND4J backend = " + Nd4j.getBackend());  // says the hardware this is running on

        Coordinator coordinator = new Coordinator();
        Thread coordinatorThread = new Thread(coordinator, "coordinator");
        coordinatorThread.start();

        System.out.println("Starting " + numWorkers + " workers ...");

        for (int i = 0; i < numWorkers; i++) {
            Worker worker = new Worker(i);
            Thread workerThread = new Thread(worker, "worker-thread-" + i);
            workerThread.start();
            System.out.println("Started worker thread " + i);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n[Shutdown] JVM is stopping, workers will close.");
        }));

        coordinatorThread.join(); // if finished every worker waits on the coordinator

        System.out.println("============== Simulation stop ==============");
        // System.exit(0);
    }

    private static void deleteDir(java.nio.file.Path path) throws java.io.IOException {
    if (!java.nio.file.Files.exists(path)) return;
    java.nio.file.Files.walk(path)
        .sorted(java.util.Comparator.reverseOrder())
        .forEach(p -> {
            try { java.nio.file.Files.delete(p); } catch (Exception ignored) {}
        });
}
}

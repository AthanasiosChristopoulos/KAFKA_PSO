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

        long start = System.nanoTime();

        Config cfg = Config.getInstance();
        int numWorkers = cfg.N_WORKERS;

        System.out.println("============== Simulation Start ======================");
        System.out.println("Starting Coordinator ...");
        System.out.println("ND4J backend = " + Nd4j.getBackend());  // says the hardware this is running on

        Coordinator coordinator = new Coordinator();
        Thread coordinatorThread = new Thread(coordinator, "coordinator");  // the coordinator starts first and then the workers
        coordinatorThread.start();

        System.out.println("Starting " + numWorkers + " workers ...");

        List<Thread> workerThreads = new ArrayList<>();


        for (int i = 0; i < numWorkers; i++) {
            Worker worker = new Worker(i);
            Thread workerThread = new Thread(worker, "worker-thread-" + i);
            workerThread.start();
            workerThreads.add(workerThread);
            System.out.println("Started worker thread " + i);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n[Shutdown] JVM is stopping, workers will close.");
        }));

        for (Thread t : workerThreads) {
            t.join();
        }

        double elapsedTime = (System.nanoTime() - start) / 1_000_000_000.0;; 
        System.out.printf("============= Training is over, ElapsedTime: %.3f =============%n", elapsedTime);

        coordinatorThread.join(); // if finished every worker waits on the coordinator
        System.out.println("============== Coordinator stopped, stopping simulation ==============");

    }

    // ===================================================================================================
    
    private static void deleteDir(java.nio.file.Path path) throws java.io.IOException {
        if (!java.nio.file.Files.exists(path)) return;
        java.nio.file.Files.walk(path)
            .sorted(java.util.Comparator.reverseOrder())
            .forEach(p -> {
                try { java.nio.file.Files.delete(p); } catch (Exception ignored) {}
            });
    }
}

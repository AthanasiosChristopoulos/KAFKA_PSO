package pso;

import java.util.ArrayList;
import java.util.List;
import java.io.InputStream;
import java.util.logging.LogManager;

public class Simulation {

    public static void main(String[] args) throws Exception {
 
        int numWorkers = Integer.parseInt(System.getenv().getOrDefault("NUM_WORKERS", "3"));

        // List<Thread> threads = new ArrayList<>();

        System.out.println("============== Simulation Start ======================");
        System.out.println("Starting Coordinator ...");

        Coordinator coordinator = new Coordinator();
        Thread coordinatorThread = new Thread(coordinator, "coordinator");
        coordinatorThread.start();
        // threads.add(coordinatorThread);

        System.out.println("Starting " + numWorkers + " workers (threads)...");

        for (int i = 0; i < numWorkers; i++) {
            Worker worker = new Worker(i);
            Thread workerThread = new Thread(worker, "worker-thread-" + i);
            workerThread.start();
            // threads.add(workerThread);
            System.out.println("Started worker thread " + i);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n[Shutdown] JVM is stopping, workers will close via their KafkaStreams hooks.");
        }));

        coordinatorThread.join(); // Wait for coordinator thread

        System.out.println("============== Simulation stop ==============");
        System.exit(1);
    }
}

package pso;

import java.util.ArrayList;
import java.util.List;
import java.io.InputStream;
import java.util.logging.LogManager;

import utils.*; 

public class Simulation {

    public static void main(String[] args) throws Exception {
        
        Config cfg = Config.getInstance();
        int numWorkers = cfg.N_WORKERS;

        System.out.println("============== Simulation Start ======================");
        System.out.println("Starting Coordinator ...");

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

        coordinatorThread.join(); 

        System.out.println("============== Simulation stop ==============");
        System.exit(1);
    }
}

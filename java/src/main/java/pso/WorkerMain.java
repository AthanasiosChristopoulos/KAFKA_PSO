package pso;

import java.util.ArrayList;
import java.util.List;

import utils.*;

public class WorkerMain {

    private static final Config cfg = Config.getInstance();

    public static void main(String[] args) throws Exception {

        cfg.FEDERATED_SETTING = true;
        Config.printExecutionInfo();
        if (args.length < 1) {
            System.err.println("Usage: WorkerMain <workerId1> <workerId2> <workerId3> ...");
            System.exit(1);
        }

        List<Thread> threads = new ArrayList<>();

        for (String arg : args) {
            int workerId = Integer.parseInt(arg);

            System.out.println("Starting Worker " + workerId + "...");

            Worker worker = new Worker(workerId, null);
            Thread t = new Thread(worker, "worker-thread-" + workerId);
            t.start();
            threads.add(t);
        }

        for (Thread t : threads) {
            t.join();
        }

        System.out.println("All requested workers finished.");
    }

}
package pso;

import java.util.ArrayList;
import java.util.List;

public class WorkerGroupMain {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: WorkerGroupMain <startWorkerId> <endWorkerId>");
            System.exit(1);
        }

        int startId = Integer.parseInt(args[0]);
        int endId = Integer.parseInt(args[1]);

        List<Thread> threads = new ArrayList<>();

        for (int i = startId; i <= endId; i++) {
            Worker worker = new Worker(i, null);
            Thread t = new Thread(worker, "worker-thread-" + i);
            t.start();
            threads.add(t);
            System.out.println("Started worker " + i);
        }

        for (Thread t : threads) {
            t.join();
        }

        System.out.println("All workers finished.");
    }
}
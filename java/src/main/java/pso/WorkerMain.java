package pso;

public class WorkerMain {
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: WorkerMain <workerId>");
            System.exit(1);
        }

        int workerId = Integer.parseInt(args[0]);

        System.out.println("Starting Worker " + workerId + " only...");

        Worker worker = new Worker(workerId, null);
        Thread workerThread = new Thread(worker, "worker-thread-" + workerId);
        workerThread.start();
        workerThread.join();

        System.out.println("Worker " + workerId + " finished.");
    }
}
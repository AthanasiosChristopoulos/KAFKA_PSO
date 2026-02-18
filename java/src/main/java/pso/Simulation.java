package pso;

import java.util.ArrayList;
import java.util.List;
import java.io.InputStream;
import java.util.logging.LogManager;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import utils.*; 

import org.nd4j.linalg.factory.Nd4j;
import java.util.Arrays;

public class Simulation {

    public static void main(String[] args) throws Exception {
        Thread.setDefaultUncaughtExceptionHandler((t, e) -> {
            System.err.println("DEFAULT uncaught in " + t.getName());
            e.printStackTrace();
        });
        // Test: =====================================================================

        // MultiLayerNetwork m = Dl4jModelFactory.createMNIST4Cnn(0);
        // float[] a = Dl4jParamUtils.modelToFlatList(m);
        // float[] b = a.clone();
        // b[0] += 0.123f;
        // Dl4jParamUtils.updateModel(m, b);
        // float[] c = Dl4jParamUtils.modelToFlatList(m);
        // System.out.println("b0 = " + b[0] + " c0 = " + c[0]);
        // System.out.println("equal = " + Arrays.equals(b, c));

        // b[302] += 0.123f;
        // Dl4jParamUtils.updateModel(m, b);
        // float[] c1 = Dl4jParamUtils.modelToFlatList(m);
        // System.out.println("b302 = " + b[302] + " c302 = " + c1[302]);
        // System.out.println("equal = " + Arrays.equals(b, c1));

        // // Stronger check: max absolute diff
        // float maxDiff = 0f;
        // int maxIdx = -1;
        // for (int i = 0; i < b.length; i++) {
        //     float d = Math.abs(b[i] - c1[i]);
        //     if (d > maxDiff) { maxDiff = d; maxIdx = i; }
        // }
        // System.out.println("maxDiff = " + maxDiff + " at idx = " + maxIdx);

        // // Optional: verify model output changed (sanity)
        // // Create a dummy MNIST-like input: shape [1,1,28,28]
        // INDArray x = Nd4j.rand(new long[]{1, 1, 28, 28});
        // INDArray y = m.output(x, false);
        // System.out.println("output shape: " + Arrays.toString(y.shape()));

        // =============================================================================================

        MultiLayerNetwork model = Dl4jModelFactory.createModel(-1, true).getFirst();

        System.out.println("OK loaded:");
        System.out.println(model.summary());

        // System.exit(0);

        // =============================================================================================

        var baseStateDir = java.nio.file.Path.of("/tmp/kstreams/");
        deleteDir(baseStateDir);
        java.nio.file.Files.createDirectories(baseStateDir);

        long start = System.nanoTime();

        Config cfg = Config.getInstance();
        int numWorkers = cfg.N_WORKERS;

        System.out.println("============== Simulation Start ======================");
        System.out.println("Starting Coordinator ...");
        // System.out.println("ND4J backend = " + Nd4j.getBackend());  // says the hardware this is running on

        System.out.println("ND4J backend: " + Nd4j.getBackend().getClass().getName());
        System.out.println("ND4J ops: " + Nd4j.getExecutioner().getClass().getName());

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

        double elapsedTimeSec = (System.nanoTime() - start) / 1_000_000_000.0; // 10^9, so this is converting to seconds    
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
}

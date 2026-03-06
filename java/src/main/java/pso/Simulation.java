package pso;

import java.util.ArrayList;
import java.util.List;
import java.io.InputStream;
import java.util.logging.LogManager;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import dl4j_models.Dl4jModelFactory;
import dl4j_models.PsoModel;

import org.deeplearning4j.nn.weights.WeightInit;

import org.deeplearning4j.nn.conf.layers.*;

import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.common.primitives.Pair;
// import org.bytedeco.cuda.global.cudart;
// import org.bytedeco.cuda.global.cudnn;

import utils.*; 

import org.nd4j.linalg.factory.Nd4j;
import java.util.Arrays;

public class Simulation {

    private static final Config cfg = Config.getInstance();

    private static String bootstrap = cfg.KAFKA_HOST;
    private static volatile long startNs = 0L;

    public static void main(String[] args) throws Exception {
        System.out.println("OK Running");
        // =================================================================================================
        // Restart the Kafka Parititions
        
        // List<String> topics;
        // if (cfg.FULLY_INFORMED || cfg.ENABLE_NEIGHBORHOODS) {
        //     topics = List.of(cfg.PBEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
        // } else {
        //     topics = List.of(cfg.GPEST_WEIGHTS_TOPIC, cfg.LOCAL_WEIGHTS_TOPIC);
        // }

        // KafkaTopicManager.recreateTopics(bootstrap, topics, 1, 1);

        // =================================================================================================

        Thread.setDefaultUncaughtExceptionHandler((t, e) -> {
            System.err.println("DEFAULT uncaught in " + t.getName());
            e.printStackTrace();
        });

        // System.out.println("Xmx = " + Runtime.getRuntime().maxMemory());
        // System.out.println("maxbytes = " + System.getProperty("org.bytedeco.javacpp.maxbytes"));
        // System.out.println("maxphysicalbytes = " + System.getProperty("org.bytedeco.javacpp.maxphysicalbytes"));


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
        // System.out.println("ND4J backend = " + Nd4j.getBackend());  // says the hardware this is running on

        System.out.println("ND4J backend: " + Nd4j.getBackend().getClass().getName());
        System.out.println("ND4J ops: " + Nd4j.getExecutioner().getClass().getName());
        System.out.println("Data type: " + Nd4j.dataType());

        Coordinator coordinator = new Coordinator(null);
        Thread coordinatorThread = new Thread(coordinator, "coordinator");  // the coordinator starts first and then the workers
        coordinatorThread.start();

        System.out.println("Starting " + numWorkers + " workers ...");

        List<Thread> workerThreads = new ArrayList<>();

        // Thread.sleep(100000000);     // 1277 MiB are loaded into VRAM before the workers even start. This is so all the CUDA/cuDNN code can be brought to memory
                                        // 1GB of allocations for Buffers
                                        
        // for (int i = 0; i < numWorkers; i++) {
        //     Worker worker = new Worker(i, null);
        //     Thread workerThread = new Thread(worker, "worker-thread-" + i);
        //     workerThread.start();
        //     workerThreads.add(workerThread);
        //     System.out.println("Started worker thread " + i);
        // }

        // for (Thread t : workerThreads) {
        //     t.join();
        // }

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

    // public static void printCudaCudnnVersions() {
    //     try {
    //         int[] v = new int[1];
    //         int rc = cudart.cudaRuntimeGetVersion(v);
    //         System.out.println("CUDA runtime: rc=" + rc + " version=" + v[0]);
    //     } catch (Throwable t) {
    //         System.out.println("CUDA runtime not available: " + t);
    //     }

    //     try {
    //         long v = cudnn.cudnnGetVersion();
    //         System.out.println("cuDNN version: " + v);
    //     } catch (Throwable t) {
    //         System.out.println("cuDNN not available: " + t);
    //     }
    // }
}

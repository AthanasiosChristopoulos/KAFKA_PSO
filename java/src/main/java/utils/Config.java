
package utils;

import io.github.cdimascio.dotenv.Dotenv;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class Config {

    private static Config instance = new Config();     // Singleton

    public final int NUM_FEATURES;      // Number of Features of Dataset
    public final int NUM_CLASSES;     // Number of Classes of Dataset
    public final int NEURAL_OUTPUT;     // Number of Classes of Dataset
    public int NUM_SAMPLES;

    public final String DATASET;
    public final String DATA_TOPIC;
    public final String TEST_TOPIC;

    public final String PBEST_WEIGHTS_TOPIC;
    public final String LOCAL_WEIGHTS_TOPIC;
    public final String GLOBAL_WEIGHTS_TOPIC;
    public final String PREDICTION_INPUT_TOPIC;
    public final String PREDICTION_OUTPUT_TOPIC;

    public final int N_WORKERS;
    public final int TRAIN_SIZE;
    public final int TEST_SIZE;
    public final int N_BATCHES;
    public final float DESIRED_ACCURACY;
    public final String SAVE_MODEL_NAME;

    public final float W_INERTIA;
    public final float W_INERTIA_START;
    public final float W_INERTIA_END;
    public final boolean ADAPTIVE_INERTIA;
    public final float C;
    public final float C1;
    public final float C2;
    public final String RUN_ID;

    public final boolean FULLY_INFORMED;
    public final boolean DEBUG_KAFKA;
    public final boolean LOG_TASK_INSTANCES;

    public final String LOSS_FUNCTION;
    public final String COMBINE_LOSS;
    public final int TOP_K_VALUE;

    public final float VMAX_FACTOR;
    public final String VMAX_CLAMPING_TYPE;

    public final float SIGNIFICANT_LOSS_DIFF;
    public final boolean FILTER_ENABLED;
    public final boolean SIMULATED_ANNEALING;

    public final int SAMPLING_CONSTANT;

    public final float MONITORING_THRESHOLD;
    public final int POINTS_PER_AXIS; 

    public final float CONVERGENCE_ALPHA;

    public final boolean ENABLE_NEIGHBORHOODS;
    public final int NEIGHBORHOOD_SIZE; 
    public final boolean INCLUDE_SELF;
    public final String NEIGHBORHOOD_TOPOLOGY;

    public final boolean INDEPENDENT_WORKER_DATA_PROCESSING;

    public Config() {
        
        Dotenv dotenv = Dotenv.configure().ignoreIfMissing().load();
        
        this.DATASET = getenv(dotenv, "DATASET", "iris");
        this.DATA_TOPIC = this.DATASET + "-input";
        this.TEST_TOPIC = this.DATASET + "-test";
        System.out.println("DATASET: " + DATASET + ", DATA_TOPIC: " + DATA_TOPIC + ", TEST_TOPIC: " + TEST_TOPIC);        
        
        this.NUM_SAMPLES = 400000;
        
        if("iris".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_IRIS", "4"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_IRIS", "3"));
        
        } else if("wine".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_WINE", "13"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_WINE", "3"));

        } else if("mnist".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_MNIST", "784"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_MNIST", "10"));

        } else if("mnist4".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_MNIST4", "784"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_MNIST4", "4"));

        } else if("susy".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_SUSY", "18"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_SUSY", "2"));

        } else if("bank".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_BANK", "21"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_BANK", "2"));

        } else if("adult".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_ADULT_INCOME", "14"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_ADULT_INCOME", "2"));

        } else if("covertype".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_COVERTYPE", "54"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_COVERTYPE", "7"));

        } else if("har".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_HAR", "54"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_HAR", "7"));

        } else if("pendigits".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_PENDIGITS", "16"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_PENDIGITS", "10"));
            this.NUM_SAMPLES = Integer.parseInt(getenv(dotenv, "NUM_SAMPLES_PENDIGITS", "400000"));

        } else if("pendigits-half".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_PENDIGITS_HALF", "16"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_PENDIGITS_HALF", "5"));
            this.NUM_SAMPLES = Integer.parseInt(getenv(dotenv, "NUM_SAMPLES_PENDIGITS_HALF", "400000"));

        } else if("winequality".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_WINEQUALITY", "12"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_WINEQUALITY", "2"));

        } else if("letter".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_LETTER", "16"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_LETTER", "26"));

        } else if("cifar3".equals(this.DATASET)) {
            this.NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_CIFAR3", "3072"));
            this.NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_CIFAR3", "3"));

        } else {
            throw new IllegalArgumentException("Invalid DATASET: " + this.DATASET);   
        }

        if(this.NUM_CLASSES == 2) {
            this.NEURAL_OUTPUT = 1;
        } else {
            this.NEURAL_OUTPUT = this.NUM_CLASSES;
        }

        this.PBEST_WEIGHTS_TOPIC = getenv(dotenv, "PBEST_WEIGHTS_TOPIC", "pbest-weights-topic");
        this.LOCAL_WEIGHTS_TOPIC = getenv(dotenv, "LOCAL_WEIGHTS_TOPIC", "local-weights-topic");
        this.GLOBAL_WEIGHTS_TOPIC = getenv(dotenv, "GLOBAL_WEIGHTS_TOPIC", "global-weights-topic");
        this.PREDICTION_INPUT_TOPIC = getenv(dotenv, "PREDICTION_INPUT_TOPIC", "iris-output");
        this.PREDICTION_OUTPUT_TOPIC = getenv(dotenv, "PREDICTION_OUTPUT_TOPIC", "iris-output");

        this.N_WORKERS = Integer.parseInt(getenv(dotenv, "N_WORKERS", "5"));
        System.out.println("N_WORKERS: " + N_WORKERS);

        this.TRAIN_SIZE = Integer.parseInt(getenv(dotenv, "TRAIN_SIZE", "30"));
        this.TEST_SIZE = Integer.parseInt(getenv(dotenv, "TEST_SIZE", "30"));
        this.DESIRED_ACCURACY = Float.parseFloat(getenv(dotenv, "DESIRED_ACCURACY", "0.9"));
        this.SAVE_MODEL_NAME = getenv(dotenv, "SAVE_MODEL_NAME", "global-model");

        this.FULLY_INFORMED = Boolean.parseBoolean(getenv(dotenv, "FULLY_INFORMED", "false"));

        if(this.FULLY_INFORMED == true) {
            this.W_INERTIA = Float.parseFloat(getenv(dotenv, "W_INERTIA_FULLY", "0.9"));
            System.out.println("Fully Informed Run");
        } else {
            this.W_INERTIA = Float.parseFloat(getenv(dotenv, "W_INERTIA_G_BEST", "0.7"));
            System.out.println("Neighborhood Best Run");
        }

        this.W_INERTIA_START = Float.parseFloat(getenv(dotenv, "W_INERTIA_START", "0.9"));
        this.W_INERTIA_END = Float.parseFloat(getenv(dotenv, "W_INERTIA_END", "0.4"));
        this.ADAPTIVE_INERTIA = Boolean.parseBoolean(getenv(dotenv, "ADAPTIVE_INERTIA", "false"));
        
        this.C = Float.parseFloat(getenv(dotenv, "C", "1.7"));
        this.C1 = Float.parseFloat(getenv(dotenv, "C1", "1.0"));
        this.C2 = Float.parseFloat(getenv(dotenv, "C2", "2.0"));

        this.RUN_ID = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")); // create new RUN_ID based on time

        this.DEBUG_KAFKA = Boolean.parseBoolean(getenv(dotenv, "DEBUG_KAFKA", "false"));
        this.LOG_TASK_INSTANCES = Boolean.parseBoolean(getenv(dotenv, "LOG_TASK_INSTANCES", "false"));

        this.LOSS_FUNCTION = getenv(dotenv, "LOSS_FUNCTION", "L2");
        this.COMBINE_LOSS = getenv(dotenv, "COMBINE_LOSS", "L2");
        this.TOP_K_VALUE = Integer.parseInt(getenv(dotenv, "TOP_K_VALUE", "5"));

        this.VMAX_FACTOR = Float.parseFloat(getenv(dotenv, "VMAX_FACTOR", "0.1"));
        this.VMAX_CLAMPING_TYPE = getenv(dotenv, "VMAX_CLAMPING_TYPE", "DIM");

        this.FILTER_ENABLED = Boolean.parseBoolean(getenv(dotenv, "FILTER_ENABLED", "false"));
        if(FILTER_ENABLED == false) {
            this.SIGNIFICANT_LOSS_DIFF = 0f;    // Essentially disables the filter
            this.N_BATCHES = Integer.parseInt(getenv(dotenv, "N_BATCHES", "30"));
            // this.N_BATCHES = Integer.parseInt(getenv(dotenv, "N_BATCHES", "30")) * 5;
        } else {
            this.SIGNIFICANT_LOSS_DIFF = Float.parseFloat(getenv(dotenv, "SIGNIFICANT_LOSS_DIFF", "0.01"));
            this.N_BATCHES = Integer.parseInt(getenv(dotenv, "N_BATCHES", "30")) * 4;
        }

        this.SIMULATED_ANNEALING = Boolean.parseBoolean(getenv(dotenv, "SIMULATED_ANNEALING", "false"));

        this.SAMPLING_CONSTANT = Integer.parseInt(getenv(dotenv, "SAMPLING_CONSTANT", "3"));

        this.MONITORING_THRESHOLD = Float.parseFloat(getenv(dotenv, "MONITORING_THRESHOLD", "40"));
        this.POINTS_PER_AXIS = Integer.parseInt(getenv(dotenv, "POINTS_PER_AXIS", "5"));

        this.CONVERGENCE_ALPHA = Float.parseFloat(getenv(dotenv, "CONVERGENCE_ALPHA", "0.01"));

        this.ENABLE_NEIGHBORHOODS = Boolean.parseBoolean(getenv(dotenv, "ENABLE_NEIGHBORHOODS", "false"));
        this.NEIGHBORHOOD_SIZE = Integer.parseInt(getenv(dotenv, "NEIGHBORHOOD_SIZE", "6"));
        this.INCLUDE_SELF = Boolean.parseBoolean(getenv(dotenv, "INCLUDE_SELF", "false"));
        this.NEIGHBORHOOD_TOPOLOGY = getenv(dotenv, "NEIGHBORHOOD_TOPOLOGY", "ring");

        this.INDEPENDENT_WORKER_DATA_PROCESSING = Boolean.parseBoolean(getenv(dotenv, "INDEPENDENT_WORKER_DATA_PROCESSING", "false"));

    }

    // ==================================================================================================================================
    
    public static Config getInstance() {
        return instance;
    }

    private static String getenv(Dotenv dotenv, String key, String def) {
        String v = System.getenv(key);      // check exported variables first
        if (v != null && !v.isBlank()) return v;

        v = dotenv.get(key);                // check .env file variables second
        if (v != null && !v.isBlank()) return v;

        return def;
    }

}

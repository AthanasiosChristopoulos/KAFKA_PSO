
package utils;

import io.github.cdimascio.dotenv.Dotenv;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class Config {

    private static Config instance = new Config();     // Singleton

    public String KAFKA_TMP_DIR;
    public String KAFKA_HOST;

    public int NUM_FEATURES;      // Number of Features of Dataset
    public int NUM_CLASSES;     // Number of Classes of Dataset
    public int NEURAL_OUTPUT;     // Number of Classes of Dataset
    public int NUM_SAMPLES;

    public String DATASET;
    public String DATA_TOPIC;
    public String TEST_TOPIC;

    public String PBEST_WEIGHTS_TOPIC;
    public String LOCAL_WEIGHTS_TOPIC;
    public String GPEST_WEIGHTS_TOPIC;
    public String PREDICTION_INPUT_TOPIC;
    public String PREDICTION_OUTPUT_TOPIC;

    public int N_WORKERS;
    public int BATCH_SIZE;
    public int TEST_BATCH_SIZE;
    public int TEST_SIZE;
    public int N_BATCHES;
    public float DESIRED_ACCURACY;
    public String SAVE_MODEL_NAME;

    public float INERTIA;
    public float INERTIA_START;
    public float INERTIA_END;
    public boolean ADAPTIVE_INERTIA;
    public float C;
    public float C1;
    public float C2;
    public String RUN_ID;

    public boolean FULLY_INFORMED;
    public boolean DEBUG_KAFKA;
    public boolean LOG_TASK_INSTANCES;

    public String LOSS_FUNCTION;
    public String COMBINE_LOSS;
    public int TOP_K_VALUE;
    public boolean NEED_LOGITS;
    public boolean NEED_PROBS;

    public float VMAX_FACTOR;
    public String VMAX_CLAMPING_TYPE;

    public float SIGNIFICANT_LOSS_DIFF;
    public float LOSS_THRESHOLD_MAX;
    public float LOSS_THRESHOLD_MIN;
    public int MONITORING_THRESHOLD_MAX;
    public int MONITORING_THRESHOLD_MIN;
    public boolean FILTER_ENABLED;
    public int PBEST_DEBOUNCE_MS;
    public boolean PREDICTION_MODELS;

    public int SAMPLING_CONSTANT;

    public float MONITORING_THRESHOLD;
    public int POINTS_PER_AXIS; 

    public float CONVERGENCE_ALPHA;

    public boolean ENABLE_NEIGHBORHOODS;
    public int NEIGHBORHOOD_SIZE; 
    public boolean INCLUDE_SELF;
    public String NEIGHBORHOOD_TOPOLOGY;

    public boolean INDEPENDENT_DATA_PROCESSING;
    public boolean GIVE_HALF_TO_SELF;

    public boolean ENABLE_LOGGING;
    public int LOGGER_LEVEL; 

    public boolean EARLY_STOPPING;
    public int MAX_NO_IMPROVEMENT_ROUNDS;
    public float CONVERGENCE_STRICTNESS_FACTOR; 

    public boolean WEIGHTS_ON_UPDATEX;

    public boolean ACCELARATION_COEFF_TIME_VARYING;
    public float C1_START; 
    public float C1_END; 
    public float C2_START; 
    public float C2_END; 
    public float C_START; 
    public float C_END; 

    public float WEIGHTS_INIT_SCALE; 
    public boolean WEIGHT_CLAMPING;
    public float WEIGHT_MAX_SCALE; 

    public boolean MEMORY_EFFICIENT;

    public int HEAD_LAYER_IDX; 
    public boolean USING_PRETRAINED_MODEL;
    public boolean TESTABLE_PRETRAINED_MODEL;

    public String REGULARIZER;
    public float LAMBDA_VALUE; 

    public long IDLE_MS; 

    public String EXPERIMENTATION_MODE;
    public String EXPERIMENTATION_DIR;

    public boolean FREEZE;
    public int FREEZE_INDEX;

    public boolean TRANSFORM_IMAGE;

    public boolean EVALUATE_PRETRAINED;
    public int MODEL_VERSION;

    public Dotenv dotenv;

    //=============================================================================================================

    public Config() {
        
        this.dotenv = Dotenv.configure().ignoreIfMissing().load();
        setAllEnviromentVariables();

    }

    //=============================================================================================================
    
    public void setAllEnviromentVariables() {
    
        KAFKA_TMP_DIR = getenv(dotenv, "cfg.KAFKA_TMP_DIR", "/tmp/kstreams");
        KAFKA_HOST = getenv(dotenv, "KAFKA_HOST", "localhost:9092");

        DATASET = getenv(dotenv, "DATASET", "iris");
        DATA_TOPIC = DATASET + "-input";
        TEST_TOPIC = DATASET + "-test";
        System.out.println("DATASET: " + DATASET + ", DATA_TOPIC: " + DATA_TOPIC + ", TEST_TOPIC: " + TEST_TOPIC);        
        
        NUM_SAMPLES = 400000;
        N_BATCHES = Integer.parseInt(getenv(dotenv, "N_BATCHES", "30"));

        if("iris".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_IRIS", "4"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_IRIS", "3"));
            N_BATCHES = 1;
        
        } else if("wine".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_WINE", "13"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_WINE", "3"));

        } else if("mnist".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_MNIST", "784"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_MNIST", "10"));

        } else if("mnist5".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_MNIST5", "784"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_MNIST5", "5"));

        } else if("fashion-mnist".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_FASHION_MNIST", "784"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_FASHION_MNIST", "10"));

        } else if("kmnist".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_KMNIST", "784"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_KMNIST", "10"));

        } else if("susy".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_SUSY", "18"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_SUSY", "2"));

        } else if("bank".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_BANK", "21"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_BANK", "2"));

        } else if("adult".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_ADULT_INCOME", "14"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_ADULT_INCOME", "2"));

        } else if("covertype".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_COVERTYPE", "54"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_COVERTYPE", "7"));

        } else if("har".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_HAR", "54"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_HAR", "7"));

        } else if("pendigits".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_PENDIGITS", "16"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_PENDIGITS", "10"));
            NUM_SAMPLES = Integer.parseInt(getenv(dotenv, "NUM_SAMPLES_PENDIGITS", "400000"));

        } else if("pendigits-half".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_PENDIGITS_HALF", "16"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_PENDIGITS_HALF", "5"));
            NUM_SAMPLES = Integer.parseInt(getenv(dotenv, "NUM_SAMPLES_PENDIGITS_HALF", "400000"));

        } else if("winequality".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_WINEQUALITY", "12"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_WINEQUALITY", "2"));

        } else if("letter".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_LETTER", "16"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_LETTER", "26"));

        } else if("cifar3".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_CIFAR3", "3072"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_CIFAR3", "3"));

        }  else if(DATASET.contains("cifar5")) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_CIFAR5", "3072"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_CIFAR5", "5"));

        } else if(DATASET.contains("cifar10")) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_CIFAR10", "3072"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_CIFAR10", "10"));
            NUM_SAMPLES = Integer.parseInt(getenv(dotenv, "NUM_SAMPLES_CIFAR10", "250000"));

        } else if("svhn".equals(DATASET)) {
            NUM_FEATURES = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_SVHN", "3072"));
            NUM_CLASSES = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_SVHN", "10"));

        } else {
            throw new IllegalArgumentException("Invalid DATASET: " + DATASET);   
        }

        if(NUM_CLASSES == 2) {
            NEURAL_OUTPUT = 1;
        } else {
            NEURAL_OUTPUT = NUM_CLASSES;
        }

        PBEST_WEIGHTS_TOPIC = getenv(dotenv, "PBEST_WEIGHTS_TOPIC", "pbest-weights-topic");
        LOCAL_WEIGHTS_TOPIC = getenv(dotenv, "LOCAL_WEIGHTS_TOPIC", "local-weights-topic");
        GPEST_WEIGHTS_TOPIC = getenv(dotenv, "GPEST_WEIGHTS_TOPIC", "global-weights-topic");
        PREDICTION_INPUT_TOPIC = getenv(dotenv, "PREDICTION_INPUT_TOPIC", "iris-output");
        PREDICTION_OUTPUT_TOPIC = getenv(dotenv, "PREDICTION_OUTPUT_TOPIC", "iris-output");

        N_WORKERS = Integer.parseInt(getenv(dotenv, "N_WORKERS", "5"));
        System.out.println("N_WORKERS: " + N_WORKERS);

        BATCH_SIZE = Integer.parseInt(getenv(dotenv, "BATCH_SIZE", "30"));
        TEST_BATCH_SIZE = Integer.parseInt(getenv(dotenv, "TEST_BATCH_SIZE", "30"));

        TEST_SIZE = Integer.parseInt(getenv(dotenv, "TEST_SIZE", "30"));
        DESIRED_ACCURACY = Float.parseFloat(getenv(dotenv, "DESIRED_ACCURACY", "0.9"));
        SAVE_MODEL_NAME = getenv(dotenv, "SAVE_MODEL_NAME", "global-model");

        FULLY_INFORMED = Boolean.parseBoolean(getenv(dotenv, "FULLY_INFORMED", "false"));

        if(FULLY_INFORMED == true) {
            INERTIA = Float.parseFloat(getenv(dotenv, "INERTIA_FULLY", "0.9"));
            System.out.println("Fully Informed Run");
        } else {
            INERTIA = Float.parseFloat(getenv(dotenv, "INERTIA_G_BEST", "0.7"));
            System.out.println("Neighborhood Best Run");
        }

        INERTIA_START = Float.parseFloat(getenv(dotenv, "INERTIA_START", "0.9"));
        INERTIA_END = Float.parseFloat(getenv(dotenv, "INERTIA_END", "0.4"));
        ADAPTIVE_INERTIA = Boolean.parseBoolean(getenv(dotenv, "ADAPTIVE_INERTIA", "false"));
        
        C = Float.parseFloat(getenv(dotenv, "C", "1.7"));
        C1 = Float.parseFloat(getenv(dotenv, "C1", "1.0"));
        C2 = Float.parseFloat(getenv(dotenv, "C2", "2.0"));

        RUN_ID = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")); // create new RUN_ID based on time

        DEBUG_KAFKA = Boolean.parseBoolean(getenv(dotenv, "DEBUG_KAFKA", "false"));
        LOG_TASK_INSTANCES = Boolean.parseBoolean(getenv(dotenv, "LOG_TASK_INSTANCES", "false"));

        LOSS_FUNCTION = getenv(dotenv, "LOSS_FUNCTION", "L2");
        COMBINE_LOSS = getenv(dotenv, "COMBINE_LOSS", "L2");
        TOP_K_VALUE = Integer.parseInt(getenv(dotenv, "TOP_K_VALUE", "5"));
        NEED_PROBS =
            "CROSS_ENTROPY".equals(LOSS_FUNCTION)
            || "MAE".equals(LOSS_FUNCTION)
            || "L2".equals(LOSS_FUNCTION)
            || "ZERO_ONE".equals(LOSS_FUNCTION)
            || "ABSOLUTE_MARGIN".equals(LOSS_FUNCTION);

        NEED_LOGITS =
            "HINGE".equals(LOSS_FUNCTION)
            || "RAMP".equals(LOSS_FUNCTION);

        VMAX_FACTOR = Float.parseFloat(getenv(dotenv, "VMAX_FACTOR", "0.1"));
        VMAX_CLAMPING_TYPE = getenv(dotenv, "VMAX_CLAMPING_TYPE", "DIM");

        // =========================================================================================================

        FILTER_ENABLED = Boolean.parseBoolean(getenv(dotenv, "FILTER_ENABLED", "false"));
        PBEST_DEBOUNCE_MS = Integer.parseInt(getenv(dotenv, "PBEST_DEBOUNCE_MS", "100"));
        PREDICTION_MODELS = Boolean.parseBoolean(getenv(dotenv, "PREDICTION_MODELS", "false"));

        if(FILTER_ENABLED == false) {
            SIGNIFICANT_LOSS_DIFF = 0f;    // Essentially disables the filter
            // N_BATCHES = Integer.parseInt(getenv(dotenv, "N_BATCHES", "30")) * 5;
            LOSS_THRESHOLD_MAX = 0f;
            LOSS_THRESHOLD_MIN = 0f;
            MONITORING_THRESHOLD_MAX = 0;
            MONITORING_THRESHOLD_MIN = 0;

        } else {
            SIGNIFICANT_LOSS_DIFF = Float.parseFloat(getenv(dotenv, "SIGNIFICANT_LOSS_DIFF", "0.01"));
            MONITORING_THRESHOLD_MAX = Integer.parseInt(getenv(dotenv, "MONITORING_THRESHOLD_MAX", "60"));
            MONITORING_THRESHOLD_MIN = Integer.parseInt(getenv(dotenv, "MONITORING_THRESHOLD_MIN", "10"));

            LOSS_THRESHOLD_MAX = Float.parseFloat(getenv(dotenv, "LOSS_THRESHOLD_MAX", "0.1"));
            LOSS_THRESHOLD_MIN = Float.parseFloat(getenv(dotenv, "LOSS_THRESHOLD_MIN", "0.005"));
        }

        // =========================================================================================================

        SAMPLING_CONSTANT = Integer.parseInt(getenv(dotenv, "SAMPLING_CONSTANT", "3"));

        MONITORING_THRESHOLD = Float.parseFloat(getenv(dotenv, "MONITORING_THRESHOLD", "40"));
        POINTS_PER_AXIS = Integer.parseInt(getenv(dotenv, "POINTS_PER_AXIS", "5"));

        CONVERGENCE_ALPHA = Float.parseFloat(getenv(dotenv, "CONVERGENCE_ALPHA", "0.01"));

        ENABLE_NEIGHBORHOODS = Boolean.parseBoolean(getenv(dotenv, "ENABLE_NEIGHBORHOODS", "false"));
        NEIGHBORHOOD_SIZE = Integer.parseInt(getenv(dotenv, "NEIGHBORHOOD_SIZE", "6"));
        INCLUDE_SELF = Boolean.parseBoolean(getenv(dotenv, "INCLUDE_SELF", "false"));
        NEIGHBORHOOD_TOPOLOGY = getenv(dotenv, "NEIGHBORHOOD_TOPOLOGY", "ring");
        if(NEIGHBORHOOD_TOPOLOGY.equals("all")) {
            ENABLE_NEIGHBORHOODS = false;
        }

        INDEPENDENT_DATA_PROCESSING = Boolean.parseBoolean(getenv(dotenv, "INDEPENDENT_DATA_PROCESSING", "false"));

        GIVE_HALF_TO_SELF = Boolean.parseBoolean(getenv(dotenv, "GIVE_HALF_TO_SELF", "false"));

        ENABLE_LOGGING = Boolean.parseBoolean(getenv(dotenv, "ENABLE_LOGGING", "true"));
        LOGGER_LEVEL = Integer.parseInt(getenv(dotenv, "LOGGER_LEVEL", "0"));

        EARLY_STOPPING = Boolean.parseBoolean(getenv(dotenv, "EARLY_STOPPING", "true"));
        MAX_NO_IMPROVEMENT_ROUNDS = Integer.parseInt(getenv(dotenv, "MAX_NO_IMPROVEMENT_ROUNDS", "30"));
        CONVERGENCE_STRICTNESS_FACTOR = Float.parseFloat(getenv(dotenv, "CONVERGENCE_STRICTNESS_FACTOR", "0.5")); 

        WEIGHTS_ON_UPDATEX = Boolean.parseBoolean(getenv(dotenv, "WEIGHTS_ON_UPDATEX", "true"));
        ACCELARATION_COEFF_TIME_VARYING = Boolean.parseBoolean(getenv(dotenv, "ACCELARATION_COEFF_TIME_VARYING", "true"));
        C1_START = Float.parseFloat(getenv(dotenv, "C1_START", "2.5")); 
        C1_END = Float.parseFloat(getenv(dotenv, "C1_END", "0.5")); 
        C2_START = Float.parseFloat(getenv(dotenv, "C2_START", "0.5")); 
        C2_END = Float.parseFloat(getenv(dotenv, "C2_END", "2.5")); 
        C_START = Float.parseFloat(getenv(dotenv, "C_START", "0.5")); 
        C_END = Float.parseFloat(getenv(dotenv, "C_END", "2.5")); 
    
        WEIGHTS_INIT_SCALE = Float.parseFloat(getenv(dotenv, "WEIGHTS_INIT_SCALE", "0.5"));
        WEIGHT_CLAMPING = Boolean.parseBoolean(getenv(dotenv, "WEIGHT_CLAMPING", "true"));
        WEIGHT_MAX_SCALE = Float.parseFloat(getenv(dotenv, "WEIGHT_MAX_SCALE", "0.5"));

        MEMORY_EFFICIENT = Boolean.parseBoolean(getenv(dotenv, "MEMORY_EFFICIENT", "false"));

        HEAD_LAYER_IDX = Integer.parseInt(getenv(dotenv, "HEAD_LAYER_IDX", "4"));
        // USING_PRETRAINED_MODEL = Boolean.parseBoolean(getenv(dotenv, "USING_PRETRAINED_MODEL", "false"));
        USING_PRETRAINED_MODEL = false;
        TESTABLE_PRETRAINED_MODEL = true;
        
        FREEZE = Boolean.parseBoolean(getenv(dotenv, "FREEZE", "true"));
        FREEZE_INDEX = Integer.parseInt(getenv(dotenv, "FREEZE_INDEX", "0"));

        REGULARIZER = getenv(dotenv, "REGULARIZER", "NONE");
        
        if(REGULARIZER.equals("L2")) {
            LAMBDA_VALUE = 1e-2f;
        
        } else if (REGULARIZER.equals("GROUP_LASSO")) {
            LAMBDA_VALUE = 1e-3f;    // or 1e-5f
        
        } else if (REGULARIZER.equals("SLOPE")) {
            LAMBDA_VALUE = 1e-3f;     // smaller because scale is large

        } else {
            LAMBDA_VALUE = 0;
        }
        
        IDLE_MS = Long.parseLong(getenv(dotenv, "IDLE_MS", "3000"));
        if(DATASET.contains("cifar") || DATASET.contains("mnist") || DATASET.contains("svhn")) {
            IDLE_MS = Long.parseLong(getenv(dotenv, "IDLE_MS", "3000")) * 4;
        }

        EXPERIMENTATION_MODE = getenv(dotenv, "EXPERIMENTATION_MODE", "N_WORKERS");
        EXPERIMENTATION_DIR = getenv(dotenv, "EXPERIMENTATION_DIR", "experimental_results_server");

        TRANSFORM_IMAGE = false;

        EVALUATE_PRETRAINED = Boolean.parseBoolean(getenv(dotenv, "EVALUATE_PRETRAINED", "false"));

        MODEL_VERSION = Integer.parseInt(getenv(dotenv, "MODEL_VERSION", "1")); 
    } 

    // ==================================================================================================================================
    
    public static Config getInstance() {
        return instance;
    }

    // ==================================================================================================================================

    public void refreshConfig() {
        if(FILTER_ENABLED == false) {
            SIGNIFICANT_LOSS_DIFF = 0f;    // Essentially disables the filter
            N_BATCHES = Integer.parseInt(getenv(dotenv, "N_BATCHES", "30"));
            // N_BATCHES = Integer.parseInt(getenv(dotenv, "N_BATCHES", "30")) * 5;
            LOSS_THRESHOLD_MAX = 0f;
            LOSS_THRESHOLD_MIN = 0f;
            MONITORING_THRESHOLD_MAX = 0;
            MONITORING_THRESHOLD_MIN = 0;

        } else {
            SIGNIFICANT_LOSS_DIFF = Float.parseFloat(getenv(dotenv, "SIGNIFICANT_LOSS_DIFF", "0.01"));
            N_BATCHES = Integer.parseInt(getenv(dotenv, "N_BATCHES", "30")) * 4;
            MONITORING_THRESHOLD_MAX = Integer.parseInt(getenv(dotenv, "MONITORING_THRESHOLD_MAX", "60"));
            MONITORING_THRESHOLD_MIN = Integer.parseInt(getenv(dotenv, "MONITORING_THRESHOLD_MIN", "10"));

            LOSS_THRESHOLD_MAX = Float.parseFloat(getenv(dotenv, "LOSS_THRESHOLD_MAX", "0.1"));
            LOSS_THRESHOLD_MIN = Float.parseFloat(getenv(dotenv, "LOSS_THRESHOLD_MIN", "0.005"));
        }

        System.out.println("MONITORING_THRESHOLD_MAX: " + MONITORING_THRESHOLD_MAX);
        System.out.println("MONITORING_THRESHOLD_MIN: " + MONITORING_THRESHOLD_MIN);

        if(NEIGHBORHOOD_TOPOLOGY.equals("all")) {
            ENABLE_NEIGHBORHOODS = false;
        }

        if(FULLY_INFORMED == true) {
            INERTIA = Float.parseFloat(getenv(dotenv, "INERTIA_FULLY", "0.9"));
            System.out.println("Fully Informed Run");
        } else {
            INERTIA = Float.parseFloat(getenv(dotenv, "INERTIA_G_BEST", "0.7"));
            System.out.println("Neighborhood Best Run");
        }

        RUN_ID = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS"));
    }

    // ==================================================================================================================================

    private static String getenv(Dotenv dotenv, String key, String def) {
        String v = System.getenv(key);      // check exported variables first
        if (v != null && !v.isBlank()) return v;

        v = dotenv.get(key);                // check .env file variables second
        if (v != null && !v.isBlank()) return v;

        return def;
    }

}

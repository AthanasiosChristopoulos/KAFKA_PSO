
package utils;

import io.github.cdimascio.dotenv.Dotenv;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class Config {

    private static Config instance = new Config();     // Singleton

    public final int NEURAL_INPUT;      // Number of Features of Dataset
    public final int NEURAL_OUTPUT;     // Number of Classes of Dataset

    public final String DATASET;
    public final String DATA_TOPIC;
    public final String TEST_TOPIC;

    public final String PBEST_WEIGHTS_TOPIC;
    public final String LOCAL_WEIGHTS_TOPIC;
    public final String GLOBAL_WEIGHTS_TOPIC;
    public final String PREDICTION_INPUT_TOPIC;
    public final String PREDICTION_OUTPUT_TOPIC;

    public final int NUM_WORKERS;
    public final int TRAIN_SIZE;
    public final int TEST_SIZE;
    public final int N_BATCHES;
    public final float DESIRED_ACCURACY;
    public final String SAVE_MODEL_NAME;

    public final float W_INERTIA;
    // public final float W_INERTIA_G_BEST;
    public final float C;
    public final float C1;
    public final float C2;
    public final String RUN_ID;

    public final boolean FULLY_INFORMED;
    public final boolean DEBUG_KAFKA;
    public final String LOSS_FUNCTION;
    public final int TOP_K_VALUE;

    public final float VMAX_FACTOR;
    public final float SIGNIFICANT_LOSS_DIFF;

    public Config() {
        
        Dotenv dotenv = Dotenv
                .configure()
                .ignoreIfMissing() 
                .load();
        
        this.DATASET = getenv(dotenv, "DATASET", "iris");
        System.out.println("DATASET: " + DATASET);        
        this.DATA_TOPIC = this.DATASET + "-input";
        this.TEST_TOPIC = this.DATASET + "-test";

        if("iris".equals(this.DATASET)) {
            this.NEURAL_INPUT = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_IRIS", "4"));
            this.NEURAL_OUTPUT = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_IRIS", "3"));
        
        } else if("wine".equals(this.DATASET)) {
            this.NEURAL_INPUT = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_WINE", "13"));
            this.NEURAL_OUTPUT = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_WINE", "3"));

        } else if("mnist".equals(this.DATASET)) {
            this.NEURAL_INPUT = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_MNIST", "784"));
            this.NEURAL_OUTPUT = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_MNIST", "10"));

        } else if("susy".equals(this.DATASET)) {
            this.NEURAL_INPUT = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_SUSY", "18"));
            this.NEURAL_OUTPUT = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_SUSY", "2"));

        } else if("bank".equals(this.DATASET)) {
            this.NEURAL_INPUT = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_BANK", "21"));
            this.NEURAL_OUTPUT = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_BANK", "2"));

        } else if("adult".equals(this.DATASET)) {
            this.NEURAL_INPUT = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_ADULT_INCOME", "14"));
            this.NEURAL_OUTPUT = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_ADULT_INCOME", "2"));

        } else if("covertype".equals(this.DATASET)) {
            this.NEURAL_INPUT = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_COVERTYPE", "54"));
            this.NEURAL_OUTPUT = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_COVERTYPE", "7"));

        } else if("har".equals(this.DATASET)) {
            this.NEURAL_INPUT = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_HAR", "54"));
            this.NEURAL_OUTPUT = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_HAR", "7"));

        }  else if("pendigits".equals(this.DATASET)) {
            this.NEURAL_INPUT = Integer.parseInt(getenv(dotenv, "NUM_FEATURES_PENDIGITS", "54"));
            this.NEURAL_OUTPUT = Integer.parseInt(getenv(dotenv, "NUM_CLASSES_PENDIGITS", "7"));

        } else {
            throw new IllegalArgumentException("Invalid DATASET: " + this.DATASET);   
        }

        this.PBEST_WEIGHTS_TOPIC = getenv(dotenv, "PBEST_WEIGHTS_TOPIC", "pbest-weights-topic");
        this.LOCAL_WEIGHTS_TOPIC = getenv(dotenv, "LOCAL_WEIGHTS_TOPIC", "local-weights-topic");
        this.GLOBAL_WEIGHTS_TOPIC = getenv(dotenv, "GLOBAL_WEIGHTS_TOPIC", "global-weights-topic");
        this.PREDICTION_INPUT_TOPIC = getenv(dotenv, "PREDICTION_INPUT_TOPIC", "iris-output");
        this.PREDICTION_OUTPUT_TOPIC = getenv(dotenv, "PREDICTION_OUTPUT_TOPIC", "iris-output");

        this.NUM_WORKERS = Integer.parseInt(getenv(dotenv, "NUM_WORKERS", "5"));
        this.TRAIN_SIZE = Integer.parseInt(getenv(dotenv, "TRAIN_SIZE", "30"));
        this.TEST_SIZE = Integer.parseInt(getenv(dotenv, "TEST_SIZE", "30"));
        this.N_BATCHES = Integer.parseInt(getenv(dotenv, "N_BATCHES", "30"));
        this.DESIRED_ACCURACY = Float.parseFloat(getenv(dotenv, "DESIRED_ACCURACY", "0.9"));
        this.SAVE_MODEL_NAME = getenv(dotenv, "SAVE_MODEL_NAME", "global-model");

        this.FULLY_INFORMED = Boolean.parseBoolean(getenv(dotenv, "FULLY_INFORMED", "false"));

        // this.W_INERTIA = Float.parseFloat(getenv(dotenv, "W_INERTIA", "0.95"));
        // this.W_INERTIA_G_BEST = Float.parseFloat(getenv(dotenv, "W_INERTIA_G_BEST", "0.7"));

        if(this.FULLY_INFORMED == true) {
            this.W_INERTIA = Float.parseFloat(getenv(dotenv, "W_INERTIA_FULLY", "0.9"));

        } else {
            this.W_INERTIA = Float.parseFloat(getenv(dotenv, "W_INERTIA_G_BEST", "0.7"));

        }

        this.C = Float.parseFloat(getenv(dotenv, "C", "1.7"));
        this.C1 = Float.parseFloat(getenv(dotenv, "C1", "1.0"));
        this.C2 = Float.parseFloat(getenv(dotenv, "C2", "2.0"));

        this.RUN_ID = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")); // create new RUN_ID based on time

        this.DEBUG_KAFKA = Boolean.parseBoolean(getenv(dotenv, "DEBUG_KAFKA", "false"));
        this.LOSS_FUNCTION = getenv(dotenv, "LOSS_FUNCTION", "L2");
        this.TOP_K_VALUE = Integer.parseInt(getenv(dotenv, "TOP_K_VALUE", "5"));

        this.VMAX_FACTOR = Float.parseFloat(getenv(dotenv, "VMAX_FACTOR", "0.1"));

        this.SIGNIFICANT_LOSS_DIFF = Float.parseFloat(getenv(dotenv, "SIGNIFICANT_LOSS_DIFF", "3.1"));

    }

    public static Config getInstance() {
        return instance;
    }

    private String getenv(Dotenv dotenv, String key, String defaultValue) {

        String value = dotenv.get(key);
        if (value != null) {
            return value;
        }
        
        System.out.println("Return Default Value");

        return defaultValue;
    }
}

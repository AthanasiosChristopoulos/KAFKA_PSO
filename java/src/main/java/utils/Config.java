
package utils;

import io.github.cdimascio.dotenv.Dotenv;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class Config {

    private static Config instance; // Singleton

    public final String DATA_TOPIC;
    public final String PBEST_WEIGHTS_TOPIC;
    public final String PREDICTION_TOPIC;
    public final String LOCAL_WEIGHTS_TOPIC;
    public final String GLOBAL_WEIGHTS_TOPIC;

    public final int NUM_WORKERS;
    public final int BATCH_SIZE;
    public final int N_BATCHES;
    public final double DESIRED_ACCURACY;
    public final String SAVE_MODEL_NAME;

    public final double W_INERTIA;
    public final double W_INERTIA_G_BEST;
    public final double C;
    public final double C1;
    public final double C2;
    public final String RUN_ID;

    public final String FULLY_INFORMED;

    public Config() {
        
        Dotenv dotenv = Dotenv
                .configure()
                .ignoreIfMissing() 
                .load();
        
        this.DATA_TOPIC = getenv(dotenv, "DATA_TOPIC", "iris-input");
        this.PBEST_WEIGHTS_TOPIC = getenv(dotenv, "PBEST_WEIGHTS_TOPIC", "pbest-weights-topic");
        this.PREDICTION_TOPIC = getenv(dotenv, "PREDICTION_TOPIC", "iris-output");
        this.LOCAL_WEIGHTS_TOPIC = getenv(dotenv, "LOCAL_WEIGHTS_TOPIC", "local-weights-topic");
        this.GLOBAL_WEIGHTS_TOPIC = getenv(dotenv, "GLOBAL_WEIGHTS_TOPIC", "global-weights-topic");

        this.NUM_WORKERS = Integer.parseInt(getenv(dotenv, "NUM_WORKERS", "30"));
        this.BATCH_SIZE = Integer.parseInt(getenv(dotenv, "BATCH_SIZE", "30"));
        this.N_BATCHES = Integer.parseInt(getenv(dotenv, "N_BATCHES", "30"));
        this.DESIRED_ACCURACY = Double.parseDouble(getenv(dotenv, "DESIRED_ACCURACY", "0.9"));
        this.SAVE_MODEL_NAME = getenv(dotenv, "SAVE_MODEL_NAME", "iris-global-model");

        this.W_INERTIA = Double.parseDouble(getenv(dotenv, "W_INERTIA", "0.95"));
        this.W_INERTIA_G_BEST = Double.parseDouble(getenv(dotenv, "W_INERTIA_G_BEST", "0.7"));
        this.C = Double.parseDouble(getenv(dotenv, "C", "1.7"));
        this.C1 = Double.parseDouble(getenv(dotenv, "C1", "1.0"));
        this.C2 = Double.parseDouble(getenv(dotenv, "C2", "2.0"));

        DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
        this.RUN_ID = LocalDateTime.now().format(fmt); 

        this.FULLY_INFORMED = getenv(dotenv, "FULLY_INFORMED", "false");
    }

    public static Config get() {
        if (instance == null) {
            instance = new Config();
        }
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

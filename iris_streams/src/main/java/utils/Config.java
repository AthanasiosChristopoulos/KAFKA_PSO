
package utils;

import io.github.cdimascio.dotenv.Dotenv;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class Config {
    
    public final String DATA_TOPIC;
    public final int BATCH_SIZE;
    public final double DESIRED_ACCURACY;
    public final String SAVE_MODEL_NAME;

    public final double C;
    public final double C1;
    public final double C2;
    public final String RUN_ID;

    public Config() {
        Dotenv dotenv = Dotenv
                .configure()
                .ignoreIfMissing()  // so it still works if .env is not there
                .load();

        // Read from .env, fallback to defaults if not present
        this.DATA_TOPIC = getenv(dotenv, "DATA_TOPIC", "iris-input");
        this.BATCH_SIZE = Integer.parseInt(getenv(dotenv, "BATCH_SIZE", "30"));
        this.DESIRED_ACCURACY = Double.parseDouble(getenv(dotenv, "DESIRED_ACCURACY", "0.9"));
        this.SAVE_MODEL_NAME = getenv(dotenv, "SAVE_MODEL_NAME", "iris-global-model");

        this.C = Double.parseDouble(getenv(dotenv, "C", "1.7"));
        this.C1 = Double.parseDouble(getenv(dotenv, "C1", "1.0"));
        this.C2 = Double.parseDouble(getenv(dotenv, "C2", "2.0"));

        DateTimeFormatter fmt = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
        runId = LocalDateTime.now().format(fmt);  
        this.RUN_ID = runId;
    }

    private String getenv(Dotenv dotenv, String key, String defaultValue) {

        // String value = System.getenv(key);
        // if (value != null) {
        //     return value;
        // }

        value = dotenv.get(key);
        if (value != null) {
            return value;
        }
        
        System.out.println("Return Default Value");

        return defaultValue;
    }
}

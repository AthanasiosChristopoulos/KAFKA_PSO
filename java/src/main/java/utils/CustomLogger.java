package utils;

import java.util.Map;
import java.util.HashMap;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.BufferedWriter;
import java.nio.file.StandardOpenOption;

public class CustomLogger {

    private Config cfg = Config.getInstance();
    public final boolean ENABLE_LOGGING = cfg.ENABLE_LOGGING;
    private final int LOGGER_LEVEL = cfg.LOGGER_LEVEL;

    public final BufferedWriter logWriter;
    public static CustomLogger coordinatorInstance;
    private static final Map<Integer, CustomLogger> workerInstances = new HashMap<>();

    private static boolean clearedLogsDir = false;

    public CustomLogger(int workerId) {

        BufferedWriter w = null;

        if(clearedLogsDir == false) {
            clearLogsDirectory();
            clearedLogsDir = true;
        }

        if (!ENABLE_LOGGING) {      // If logging is enabled, clear the directory 
                                    // and return without creating new log files
            this.logWriter = null;
            return;
        }

        if(workerId != -1) {

            try {
                Files.createDirectories(Paths.get("logs"));
                w = Files.newBufferedWriter(
                        Paths.get("logs/worker_" + workerId + ".log"),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.TRUNCATE_EXISTING,
                        StandardOpenOption.WRITE
                );
            } catch (IOException e) {
                e.printStackTrace();
            }

        } else {

            try {
                Files.createDirectories(Paths.get("logs"));
                w = Files.newBufferedWriter(
                        Paths.get("logs/coordinator.log"),
                        StandardOpenOption.CREATE,
                        StandardOpenOption.TRUNCATE_EXISTING,
                        StandardOpenOption.WRITE
                );
            } catch (IOException e) {
                e.printStackTrace();
            }            
        } 
        
        this.logWriter = w;

    } 
    
    //=====================================================================================\

    public static CustomLogger getInstanceForCoordinator() {
        if(coordinatorInstance == null) {
            coordinatorInstance = new CustomLogger(-1);
        }
        return coordinatorInstance;
    }

    //=====================================================================================\

    public static CustomLogger getWorkerInstance(int workerId) {
        CustomLogger logger = workerInstances.get(workerId);

        if (logger == null) {
            logger = new CustomLogger(workerId);
            workerInstances.put(workerId, logger);
        }

        return logger;
    }

    //=====================================================================================\

    private static void clearLogsDirectory() {
        try {
            Path logsDir = Paths.get("logs");

            if (Files.exists(logsDir)) {
                Files.list(logsDir)
                    .filter(Files::isRegularFile)
                    .forEach(path -> {
                        try {
                            Files.delete(path);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    });
            } else {
                Files.createDirectories(logsDir);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //=====================================================================================

    public boolean isEnabled(int log_level) {
        return ENABLE_LOGGING && LOGGER_LEVEL <= log_level;
    }

    //=====================================================================================

    public void log(String msg) {
        // if (logWriter == null || !ENABLE_LOGGING) return;

        // if (logWriter == null || !ENABLE_LOGGING || LOGGER_LEVEL > log_level) return;
            // Logger Level: 
            //  0 => causes performance issues, recurring log, not critical 
            //  1 => uncommon, recurring log message (pBest update)
            //  2 => errors / final report / critical / only once logs

        try {
            logWriter.write(msg);
            logWriter.newLine();
            logWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

package utils;

import java.util.Map;
import java.util.HashMap;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.BufferedWriter;
import java.nio.file.StandardOpenOption;

public class CustomLogger {

    public final BufferedWriter logWriter;

    public CustomLogger(int workerId) {
        
        BufferedWriter w = null;
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
        
        this.logWriter = w;
    } 

    public void log(String msg) {
        if (logWriter == null) return;
        try {
            logWriter.write(msg);
            logWriter.newLine();
            logWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

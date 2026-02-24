package experimentation;
import utils.*;
import state.*;

import java.io.FileWriter;
import java.util.List;

public class Experimentation {
    public static void main(String[] args) throws Exception {

        List<Integer> workersList = List.of(2, 4, 6);

        try (FileWriter w = new FileWriter("experimental_results/results.csv")) {
            w.write("N_WORKERS,TOTAL_ELAPSED,COORD_ELAPSED,LAST_WORKER_ELAPSED,GBEST_ACC,GBEST_LOSS\n");

            for (int n : workersList) {
                Config cfg = Config.getInstance();
                cfg.N_WORKERS = n; // or better: create a Config copy per run

                ExperimentResult r = SimulationRunner.runOnce(cfg);

                double coordElapsed = r.getCoordinator() != null ? r.getCoordinator().getElapsedSec() : Double.NaN;

                w.write(String.format(
                        "%d,%.3f,%.3f,%.3f,%.6f,%.6f\n",
                        r.getWorkers(),
                        r.getTotalElapsedSec(),
                        coordElapsed,
                        r.lastWorkerElapsedSec(),
                        r.getCoordinator() != null ? r.getCoordinator().getGlobalBestAcc() : Double.NaN,
                        r.getCoordinator() != null ? r.getCoordinator().getGlobalBestLoss() : Double.NaN
                ));
                w.flush();
            }
        }
    }
}
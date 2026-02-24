
package state;

import java.util.concurrent.atomic.AtomicBoolean;

import utils.*;

public class CoordinatorControl {

    private static final Config cfg = Config.getInstance();
    private static final int N_WORKERS = cfg.N_WORKERS;
    private static int count = N_WORKERS;
    private static float bestGlobalModelAccuracy = -1f;
    private static float bestTrainingAccuracy = -1f;

    private static final CoordinatorControl instance = new CoordinatorControl();

    private final AtomicBoolean[] workerStopRequested = new AtomicBoolean[N_WORKERS];
    private final AtomicBoolean stopRequestedFinal = new AtomicBoolean(false);

    // =================================================================================================

    private CoordinatorControl() {
        for (int i = 0; i < N_WORKERS; i++) {
            workerStopRequested[i] = new AtomicBoolean(false);
        }
    }
    
    // =================================================================================================

    public void requestStop(int workerId) {
        if (workerId < 0 || workerId >= N_WORKERS) {
            throw new IllegalArgumentException("Invalid workerId: " + workerId);
        }

        if (workerStopRequested[workerId].compareAndSet(false, true)) {
            count = count - 1;
            if (count <= 0) {
                stopRequestedFinal.set(true);   // Set after every worker requests stop. N
            }
        }
    }

    // =================================================================================================

    public void requestStopFinal() {    // Is set after every worker requests stop. Not used by coordinator, because DESIRED_ACCURACY = 100%
        stopRequestedFinal.set(true);
    }

    // =================================================================================================

    public boolean isStopRequested(int workerId) {
        if (workerId < 0 || workerId >= N_WORKERS) {
            return stopRequestedFinal.get();
        }
        return stopRequestedFinal.get() || workerStopRequested[workerId].get();
    }

    // =================================================================================================

    public static CoordinatorControl getInstance() {
        return instance;
    }

    public static float getBestGlobalModelAccuracy() {
        return bestGlobalModelAccuracy;
    }

    public static void setBestGlobalModelAccuracy(float bestGlobalModelAccuracy) {
        CoordinatorControl.bestGlobalModelAccuracy = bestGlobalModelAccuracy;
    }

    public static float getBestTrainingAccuracy() {
        return bestTrainingAccuracy;
    }

    public static void setBestTrainingAccuracy(float bestTrainingAccuracy) {
        CoordinatorControl.bestTrainingAccuracy = bestTrainingAccuracy;
    }

    // =================================================================================================
    public synchronized void resetForNewRun() {
        this.stopFinal = false;
        // if you have per-worker stop flags, clear them too
        // e.g. stopRequestedWorkers.clear();

        // also reset best metrics if you reuse the same singleton
        this.bestTrainingAccuracy = 0.0;
        this.bestGlobalModelAccuracy = 0.0;
        // reset any other cached state you have
    }

        // =================================================================================================

}

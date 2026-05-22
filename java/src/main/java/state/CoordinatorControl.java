
package state;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import utils.*;

public class CoordinatorControl {

    private static final Config cfg = Config.getInstance();
    private static int count = cfg.N_WORKERS;
    private static float bestGlobalModelAccuracy = -1f;
    private static float bestGlobalModelLoss= 10000f;
    private static float bestTrainingAccuracy = -1f;
    private static float pretrainedAccuracy = -1f;

	private static final CoordinatorControl instance = new CoordinatorControl();

    private AtomicBoolean[] workerStopRequested = new AtomicBoolean[cfg.N_WORKERS];
    private AtomicBoolean stopRequestedFinal = new AtomicBoolean(false);

    public final List<AccuracyPoint> accuracyValues = new ArrayList<>();
        
    public boolean processedAtLeastOne = false;
    public double avgWeightsMessageDelayMsSum = 0;

    // =================================================================================================

    private CoordinatorControl() {
        
        for (int i = 0; i < cfg.N_WORKERS; i++) {
            workerStopRequested[i] = new AtomicBoolean(false);
        }
    }
    
    // =================================================================================================

    public void requestStop(int workerId) {
        if (workerId < 0 || workerId >= cfg.N_WORKERS) {
            throw new IllegalArgumentException("Invalid workerId: " + workerId);
        }

        if (workerStopRequested[workerId].compareAndSet(false, true)) {
            count = count - 1;
            if (count <= 0) {
                stopRequestedFinal.set(true);   
            }
        }
    }

    // =================================================================================================

    public void requestStopFinal() {    // Is set after every worker requests stop or once by coordinator directly
        stopRequestedFinal.set(true);
    }

    // =================================================================================================

    public boolean isStopRequested(int workerId) {
        if (workerId < 0 || workerId >= cfg.N_WORKERS) {
            return stopRequestedFinal.get();
        }
        return stopRequestedFinal.get() || workerStopRequested[workerId].get();
    }

    // =================================================================================================

    public static CoordinatorControl getInstance() {
        return instance;
    }

    // =================================================================================================

    public float getPretrainedAccuracy() {
        return pretrainedAccuracy;
    }
    // =================================================================================================

    public void setPretrainedAccuracy(float pretrainedAccuracy) {
        CoordinatorControl.pretrainedAccuracy = pretrainedAccuracy;
    }
    
    // =================================================================================================

    public float getBestGlobalModelAccuracy() {
        return bestGlobalModelAccuracy;
    }

    // =================================================================================================

    public void setBestGlobalModelAccuracy(float bestGlobalModelAccuracy) {
        CoordinatorControl.bestGlobalModelAccuracy = bestGlobalModelAccuracy;
    }

    // =================================================================================================

    public float getBestGlobalModelLoss() {
		return bestGlobalModelLoss;
	}

    // =================================================================================================

    public void setBestGlobalModelLoss(float bestGlobalModelLoss) {
		CoordinatorControl.bestGlobalModelLoss = bestGlobalModelLoss;
	}

    // =================================================================================================

	public float getBestTrainingAccuracy() {
        return bestTrainingAccuracy;
    }

    // =================================================================================================

    public void setBestTrainingAccuracy(float bestTrainingAccuracy) {
        CoordinatorControl.bestTrainingAccuracy = bestTrainingAccuracy;
    }

    // =================================================================================================

    public synchronized void resetForNewRun(int nWorkers) {
        
        stopRequestedFinal.set(false);
        bestGlobalModelAccuracy = -1f;
        bestTrainingAccuracy = -1f;
        count = nWorkers;

        workerStopRequested = new AtomicBoolean[nWorkers];
        for (int i = 0; i < nWorkers; i++) workerStopRequested[i] = new AtomicBoolean(false);
        
        processedAtLeastOne = false;
    }
}

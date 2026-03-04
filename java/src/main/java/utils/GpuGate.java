package utils;

import java.util.concurrent.Semaphore;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import dl4j_models.PsoModel;

public class GpuGate {
    public static final Semaphore GPU_SEMAPHORE = new Semaphore(1, true);

    public static INDArray outputExclusive(PsoModel model, INDArray x, int workerId) {
        boolean acquired = false;
        try {
            GPU_SEMAPHORE.acquire();
            acquired = true;
            // System.out.println("[Worker " + workerId + "] aqcuired GPU");
            INDArray y = model.output(x, false);

            // Important: make sure GPU kernels finish before we "release"
            // otherwise the next worker can pile on while GPU is still busy.
            Nd4j.getExecutioner().commit();

            // System.out.println("[Worker " + workerId + "] leaving GPU");

            return y;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while waiting for GPU", e);
        } finally {
            if (acquired) GPU_SEMAPHORE.release();
        }
    }
}
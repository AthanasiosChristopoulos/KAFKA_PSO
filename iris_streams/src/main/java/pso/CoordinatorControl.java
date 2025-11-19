package pso;

import java.util.concurrent.atomic.AtomicBoolean;

public class CoordinatorControl {

    private final AtomicBoolean stopRequested = new AtomicBoolean(false);

    public void requestStop() {
        stopRequested.set(true);
    }

    public boolean isStopRequested() {
        return stopRequested.get();
    }
}

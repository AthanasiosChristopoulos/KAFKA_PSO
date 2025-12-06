package state;

import java.util.concurrent.atomic.AtomicBoolean;

public class CoordinatorControl {

    private final AtomicBoolean stopRequested = new AtomicBoolean(false);
    private static final CoordinatorControl instance = new CoordinatorControl();

    private CoordinatorControl() {

    }

    public void requestStop() {
        stopRequested.set(true);
    }

    public boolean isStopRequested() {
        return stopRequested.get();
    }

    public static CoordinatorControl getInstance() {
        return instance;
    }
}

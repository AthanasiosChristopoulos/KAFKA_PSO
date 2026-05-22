package experimentation;

import utils.Config;

public final class FilterStrength {

    public enum Level {
        OFF(0),
        EASY(1),
        MEDIUM(2),
        HARD(3);

        public final int code;
        Level(int code) { this.code = code; }

        public static Level fromCode(int code) {
            for (Level l : values()) if (l.code == code) return l;
            throw new IllegalArgumentException("Unknown strength code: " + code);
        }
    }

    // =======================================================================================

    public static final class Preset {
        public final float lossMin;
        public final float lossMax;
        public final int pbestDebounceMs;
        public final int monitoringMin;
        public final int monitoringMax;

        public Preset(float lossMin, float lossMax, int pbestDebounceMs, int monitoringMin, int monitoringMax) {
            this.lossMin = lossMin;
            this.lossMax = lossMax;
            this.pbestDebounceMs = pbestDebounceMs;
            this.monitoringMin = monitoringMin;
            this.monitoringMax = monitoringMax;
        }
    }

    // =====================================================================================
    public static Preset preset(Level level) {
        return switch (level) {
            case OFF    -> new Preset(0.0f, 0.0f, 0, 0, 0);
            case EASY   -> new Preset(0.01f, 0.10f, 25, 5, 20);
            case MEDIUM -> new Preset(0.01f, 0.15f, 50, 10, 40);
            case HARD   -> new Preset(0.05f, 0.15f, 75, 15, 60);
        };
    }

    // =============================================================================================

    public static void apply(Config cfg, Level level) {

        if(level == Level.OFF) {
            cfg.FILTER_ENABLED = false;
        } else {
            cfg.FILTER_ENABLED = true;
        }
        

        Preset p = preset(level);
        cfg.LOSS_THRESHOLD_MIN = p.lossMin;
        cfg.LOSS_THRESHOLD_MAX = p.lossMax;
        cfg.PBEST_DEBOUNCE_MS = p.pbestDebounceMs;
        cfg.MONITORING_THRESHOLD_MIN = p.monitoringMin;
        cfg.MONITORING_THRESHOLD_MAX = p.monitoringMax;
    }

    private FilterStrength() {}
}
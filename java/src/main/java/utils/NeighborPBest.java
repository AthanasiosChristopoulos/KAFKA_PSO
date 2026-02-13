package utils;

public class NeighborPBest {
    public final float[] pBest;
    public final float accuracy;
    public final int workerId;

    public NeighborPBest(float[] pBest, float accuracy, int workerId) {
        this.pBest = pBest;
        this.accuracy = accuracy;
        this.workerId = workerId;
    }

}
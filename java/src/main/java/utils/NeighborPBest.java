package utils;

public class NeighborPBest {
    public final float[] pBest;
    public final float accuracy;

    public NeighborPBest(float[] pBest, float accuracy) {
        this.pBest = pBest;
        this.accuracy = accuracy;
    }

}
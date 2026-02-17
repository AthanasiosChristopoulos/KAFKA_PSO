package utils;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

public final class ParamSlices {

    public static int headFlatIndex(MultiLayerNetwork model, int headStartLayerIdx) {
        int off = 0;
        for (int i = 0; i < headStartLayerIdx; i++) {
            off += model.getLayer(i).numParams();
        }
        return off;
    }

    public static int totalParams(MultiLayerNetwork model) {
        return (int) model.numParams();
    }
}

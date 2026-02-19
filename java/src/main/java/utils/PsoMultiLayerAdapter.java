package utils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.Map;

public final class PsoMultiLayerAdapter implements PsoModel {
    private final MultiLayerNetwork model;

    public PsoMultiLayerAdapter(MultiLayerNetwork model) { this.model = model; }

    @Override public INDArray params() { return model.params(); }

    @Override public void setParamsFromFlat(float[] w) {
        // in-place, no allocations
        model.params().data().setData(w);
    }

    @Override public float[] getParamsAsFlat() {
        return model.params().toFloatVector();
    }

    @Override public INDArray output(INDArray X) {
        return model.output(X, false);
    }

    @Override public int numParams() {
        return (int) model.numParams();
    }

    @Override public int numOutputs() {
        // MultiLayerNetwork output is a single INDArray
        return 1;
    }

    @Override public MultiLayerNetwork asMultiLayerNetwork(){ return model; }

    @Override public String summary() { return model.summary(); }

    @Override public Map<String, INDArray> paramTable() { return model.paramTable(); }

    @Override public boolean isCnn() {
        org.deeplearning4j.nn.conf.layers.Layer l0 =
            model.getLayerWiseConfigurations().getConf(0).getLayer();
        return l0 instanceof org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
    }

    @Override public boolean isNhWC() { return false; }
    
    @Override public INDArray output(INDArray X, boolean training) {
        return model.output(X, training);
    }

}
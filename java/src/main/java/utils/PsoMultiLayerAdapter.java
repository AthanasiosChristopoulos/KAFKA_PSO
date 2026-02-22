package utils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.Map;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;

public final class PsoMultiLayerAdapter implements PsoModel {
    private final MultiLayerNetwork model;
    private boolean nhwc = false;

    public PsoMultiLayerAdapter(MultiLayerNetwork model) { 
        this.model = model; 
    }
    public PsoMultiLayerAdapter(MultiLayerNetwork model, boolean nhwc) { 
        this.model = model; 
        this.nhwc = nhwc;
    }

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

    @Override public MultiLayerNetwork asMultiLayerNetwork() { return model; }

    @Override public String summary() { return model.summary(); }

    @Override public Map<String, INDArray> paramTable() { return model.paramTable(); }

    @Override public boolean isCnn() {
        Layer l = model.getLayerWiseConfigurations().getConf(0).getLayer();
        while (true) {
            if (l instanceof FrozenLayer) {
                l = ((FrozenLayer) l).getLayer();
                continue;
            }
            if (l instanceof BaseWrapperLayer) { 
                l = ((BaseWrapperLayer) l).getUnderlying();
                continue;
            }
            break;
        }

        return (l instanceof ConvolutionLayer); // if found a conv layer then it cnn
    }

    @Override public boolean isNhWC() { return nhwc; }
    
    @Override public INDArray output(INDArray X, boolean training) {
        return model.output(X, training);
    }

    @Override public void close() {
        model.close(); // if available in your DL4J version
    }
}
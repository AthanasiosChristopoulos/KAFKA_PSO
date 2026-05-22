package dl4j_models;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.graph.ComputationGraph;
import java.util.Map;

public final class PsoGraphAdapter implements PsoModel {
    private final ComputationGraph model;

    public PsoGraphAdapter(ComputationGraph model) { this.model = model; }

    @Override public INDArray params() { return model.params(); }

    @Override public void setParamsFromFlat(float[] w) {
        model.params().data().setData(w);
    }

    @Override public float[] getParamsAsFlat() {
        return model.params().toFloatVector();
    }

    @Override public INDArray output(INDArray X) {
        return model.outputSingle(false, X);
    }

    @Override public int numParams() {
        return (int) model.numParams();
    }

    @Override public int numOutputs() {
        return model.getNumOutputArrays();
    }

    @Override public ComputationGraph asComputationGraph() { return model; }

    @Override public String summary() { return model.summary(); }

    @Override public Map<String, INDArray> paramTable() { return model.paramTable(); }

    @Override public boolean isCnn() { return true; }

    @Override public boolean isNhWC() { return true; }

    @Override public INDArray output(INDArray X, boolean training) {
        return model.outputSingle(training, X);
    }

    @Override public void close() {
        model.close(); 
        model.clear();
    }
}

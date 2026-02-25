package dl4j_models;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.Map;

public interface PsoModel {
    INDArray params();                 // view of internal param buffer
    void setParamsFromFlat(float[] w); // in-place update
    float[] getParamsAsFlat();         // convenience
    INDArray output(INDArray X);       // forward pass -> probabilities/logits
    int numParams();
    int numOutputs();                  // optional, for sanity checks

    default MultiLayerNetwork asMultiLayerNetwork() { return null; }
    default ComputationGraph asComputationGraph() { return null; }
    default void close() { /* no-op by default */ }
    
    String summary();
    Map<String, INDArray> paramTable();

    boolean isCnn();
    boolean isNhWC(); 

    INDArray output(INDArray X, boolean training);



}
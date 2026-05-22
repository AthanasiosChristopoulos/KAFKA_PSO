package dl4j_models;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.Map;

public interface PsoModel {
    INDArray params();      
    void setParamsFromFlat(float[] w);
    float[] getParamsAsFlat();   
    INDArray output(INDArray X);    
    int numParams();
    int numOutputs();           

    default MultiLayerNetwork asMultiLayerNetwork() { return null; }
    default ComputationGraph asComputationGraph() { return null; }
    default void close() { }
    
    String summary();
    Map<String, INDArray> paramTable();

    boolean isCnn();
    boolean isNhWC(); 

    INDArray output(INDArray X, boolean training);



}
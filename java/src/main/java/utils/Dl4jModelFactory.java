package utils;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Dl4jModelFactory {
        
	private static final Config cfg = Config.getInstance();
	private static final String DATA_TOPIC = cfg.DATA_TOPIC;
    public static final int NEURAL_INPUT = cfg.NEURAL_INPUT;
    public static final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;

	public static MultiLayerNetwork createModel() {
		// System.out.println("DATA_TOPIC: " + DATA_TOPIC);

		if("iris-input".equals(DATA_TOPIC)) {
				return createIrisModel();

		} else if ("wine-input".equals(DATA_TOPIC)) {
				return createWineModel();
		} else {
            throw new IllegalArgumentException("Invalid DATA_TOPIC: " + DATA_TOPIC);
		}
	}

	// ======================================================================================================================
	// Iris Dataset Model Architecture 

	public static MultiLayerNetwork createIrisModel() {
		System.out.println("Using Iris Model");
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123) // or pass seed from outside
				.list()
				.layer(new DenseLayer.Builder() // Hidden Layer 1 (with input Layer)
						.nIn(NEURAL_INPUT)
						.nOut(16)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder() // Hidden Layer 2
						.nIn(16)
						.nOut(16)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder() // Output Layer 
						.nIn(16)
						.nOut(NEURAL_OUTPUT)
						.lossFunction(LossFunctions.LossFunction.MCXENT)  // is used only for model.fit(...). Ignore it
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init(); // sets the random weights 
		return model;
	} 

	// Number of weights in the network calculation:  
        // For Hidden Layer 1   => 4 * 16 + 16 (Bias) 
        // For Hidden Layer 2   => 16 * 16 + 16
        // For Output Layer     => 16 * 3 + 3
        // 403 weights all in all

	// ======================================================================================================================
	// Wine Dataset Model Architecture 

	public static MultiLayerNetwork createWineModel() {
		System.out.println("Using Wine Model");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(NEURAL_INPUT)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(32)
                        .nOut(16)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(16)
                        .nOut(NEURAL_OUTPUT)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
	} 
	// Number of weights in the network calculation:  
        // For Hidden Layer 1   => 13 * 32 + 32 (Bias) 
        // For Hidden Layer 2   => 32 * 16 + 16
        // For Output Layer     => 16  * 3 + 3
        // 403 weights all in all


}

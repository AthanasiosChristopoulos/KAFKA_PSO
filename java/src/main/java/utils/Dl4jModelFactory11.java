package utils;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Dl4jModelFactory {

        public static MultiLayerNetwork createIrisModel() {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(123) // or pass seed from outside
                        .list()
                        .layer(new DenseLayer.Builder() // Hidden Layer 1 (with input Layer)
                                .nIn(4)
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
                                .nOut(3)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)  // is used only for model.fit(...). Ignore it
                                .activation(Activation.SOFTMAX)
                                .build())
                        .build();

                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init(); // sets the random weights 
                return model;
        } 
    
        public static MultiLayerNetwork createIrisModel() {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(123) // or pass seed from outside
                        .list()
                        .layer(new DenseLayer.Builder() // Hidden Layer 1 (with input Layer)
                                .nIn(4)
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
                                .nOut(3)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)  // is used only for model.fit(...). Ignore it
                                .activation(Activation.SOFTMAX)
                                .build())
                        .build();

                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init(); // sets the random weights 
                return model;
        } 
}

// Number of weights in the network calculation:  
        // For Hidden Layer 1   => 4 * 16 + 16 (Bias) 
        // For Hidden Layer 2   => 16 * 16 + 16
        // For Output Layer     => 16 * 3 + 3
        // 403 weights all in all
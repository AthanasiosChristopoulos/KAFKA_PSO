package utils;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;

import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.Adam;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;

import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.nd4j.linalg.learning.config.NoOp;

public class Dl4jModelFactory {
        
	private static final Config cfg = Config.getInstance();
	private static final String DATASET = cfg.DATASET;
    public static final int NUM_FEATURES = cfg.NUM_FEATURES;
    public static final int NUM_CLASSES = cfg.NUM_CLASSES;
    public static final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;
    private static final float WEIGHTS_INIT_SCALE = cfg.WEIGHTS_INIT_SCALE;

	public static final boolean printModel = false;

	public static MultiLayerNetwork createModel(int workerId, boolean preTrained) {
		// System.out.println("DATASET: " + DATASET);

		if("iris".equals(DATASET)) {
			return createIrisModel(workerId);

		} else if ("wine".equals(DATASET)) {
			return createWineModel(workerId);

		} else if ("mnist".equals(DATASET)) {
			// return createMNISTModelMLP(workerId);
			// return createMNISTModelMLPSimple_0(workerId);
			// return createMNISTModelMLPSimple_1(workerId);
			// return createMNISTModelMLPSimple_2(workerId);
			// return createMNISTCnn(workerId);
			// return createMNIST4Cnn_New(workerId);
			// return createMNISTCnn_New_2(workerId);
			// return createMNISTModelCNNHeavy(workerId);

			if(preTrained) {
				// return pretrainedModelLeNet();
				return pretrainedModelFashionMNIST();
			} else {
				// return createMNIST_CNN_PretrainedLeNet(workerId);
				return createMNIST_CNN_PretrainedFashionMNIST(workerId);
			}

		} else if ("mnist4".equals(DATASET)) {	// Forward pass cost: CPU => 200ms / GPU => 30ms  
			// return createMNISTModelMLP(workerId);
			// return createMNISTModelMLPSimple_1(workerId);
			return createMNISTModelMLPSimple_2(workerId);
			// return createMNIST4Cnn(workerId);	// 70ms forward pass
			// return createMNIST4Cnn_Simple(workerId);	// 25ms forward pass on average
			// return createMNIST4MLP(workerId);
			// return createMNIST4MLP_Reduced(workerId);
			// return createMNIST4Cnn_New(workerId);			// this costs on forward pass much more time (60ms)
			// return createMNIST4Cnn_New_Simpler(workerId);
			// return createMNIST4Cnn_New_2(workerId);			// this costs a lot less on forwaard pass and gets the same performance (22ms)


		} else if ("fashion_mnist".equals(DATASET)) {
			// return createMNISTModelMLPSimple_2(workerId);
			// return createMNISTModelMLPSimple_1(workerId);
			return createMNIST4Cnn_New_Simpler(workerId);
			// return createMNISTCnn_New_2(workerId);
			// return createMNISTCnn(workerId);
			// return createMNIST4Cnn_New(workerId);

		} else if ("susy".equals(DATASET)) {
			// return createSUSYModel_SOFTMAX(workerId);
			return createSUSYModel(workerId);

		} else if ("bank".equals(DATASET)) {
			// return createBankModel(workerId);
			return createBankModel40K(workerId);

		} else if ("adult".equals(DATASET)) {
			return createAdultModel(workerId);

		} else if ("covertype".equals(DATASET)) {
			return createCovertypeModel(workerId);

		} else if ("har".equals(DATASET)) {
			return createHarModel(workerId);

		} else if ("pendigits".equals(DATASET) || "pendigits-half".equals(DATASET)) {
			// return createPendigitsModel(workerId);	// forward pass cost: CPU = 10ms / GPU = 3ms
			return createPendigitsModelTanh(workerId);
			// return createPendigitsModelSmaller(workerId);
			// return createPendigitsModelSmaller_2(workerId);
			// return createPendigitsModelSmaller_3(workerId);

		} else if ("winequality".equals(DATASET)) {
			return createWineQualityModel(workerId);

		} else if ("letter".equals(DATASET)) {
			return createLetterModel(workerId);
			// return createLetterModel70K(workerId);
		} else if ("cifar3".equals(DATASET)) {
			// return createCifar3Model_PSO_Simple(workerId);
			// return createCifar3Model(workerId);
			// return createCifar3Model_New(workerId);	
			// return createCifar3Model_New_Simpler(workerId);
			// return createCifar3Model_New_Simpler_2(workerId);
			return createCifar3Model_New_Simpler_3(workerId);
			// return createCifar3Model_New_Simpler_4(workerId);
			// return createMNIST4Cnn_New_Simpler(workerId);
		} else {
            throw new IllegalArgumentException("Invalid DATASET: " + DATASET);
		}
	}

	// ======================================================================================================================

	public static MultiLayerNetwork pretrainedModelFashionMNIST() {
		try {
			// File f = new File("pretrained_models/fmnist_base_plus_head.h5");
			File f = new File("pretrained_models/mnist_base_plus_head.h5");
			
			if (!f.exists()) {
				throw new IllegalStateException("Missing pretrained Keras model: " + f.getAbsolutePath());
			}

			MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(
					f.getAbsolutePath(),
					false   // enforceTrainingConfig = false (not using Keras optimizer config)
			);

			return model;

		} catch (Exception e) {
			throw new RuntimeException("Failed to import Fashion-MNIST Keras .h5 model", e);
		}
	}

	// ======================================================================================================================

	// public static MultiLayerNetwork createMNIST_CNN_PretrainedFashionMNIST(int workerId) {

	// 	MultiLayerNetwork pretrained = pretrainedModelFashionMNIST();

	// 	// We are going to replace the last TWO trainable layers: Dense(64) and Dense(10)
	// 	int removeCount = removeCountForLastNTrainableLayers(pretrained, 2);

	// 	FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
	// 			.updater(new NoOp())   // PSO controls weights, not SGD
	// 			.build();

	// 	// After removing Dense(64) and Dense(10), the layer feeding the head is the Flatten output.
	// 	// In your architecture: 28x28 -> conv valid -> pool -> conv valid -> pool -> flatten = 32*5*5 = 800
	// 	// If your conv/pool settings match: 28->26->13->11->5 => channels 32 => 32*5*5 = 800
	// 	final int flattenDim = 32 * 5 * 5;  // 800

	// 	MultiLayerNetwork tl = new TransferLearning.Builder(pretrained)
	// 			.fineTuneConfiguration(ftc)
	// 			.removeLayersFromOutput(removeCount)
	// 			.addLayer(new DenseLayer.Builder()
	// 					.nIn(flattenDim)
	// 					.nOut(32)
	// 					.activation(Activation.RELU) // or TANH for PSO smoothness
	// 					.build())
	// 			.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
	// 					.nIn(32)
	// 					.nOut(NUM_CLASSES)           // MNIST=10 or MNIST4=4 depending on cfg
	// 					.activation(Activation.SOFTMAX)
	// 					.build())
	// 			.build();

	// 	return tl;
	// }

	// ======================================================================================================================

	public static MultiLayerNetwork createMNIST_CNN_PretrainedFashionMNIST(int workerId) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = pretrainedModelFashionMNIST();

		// ============================================================================
		// DL4J needs a FineTuneConfiguration to define the updater (Adam, SGD, learning rate )
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.updater(new NoOp())   // <-- prevents optimizer assumptions
				.build();

		// From summary
		final int flattenDim = 32 * 5 * 5;  // 800
		MultiLayerNetwork tl = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(2)
				.addLayer(new DenseLayer.Builder()
						.nIn(flattenDim)
						.nOut(64)
						.activation(Activation.RELU) // or TANH for PSO smoothness
						.build())
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(64)
						.nOut(NUM_CLASSES)           // MNIST=10 or MNIST4=4 depending on cfg
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		return tl;
	}
	// ======================================================================================================================

	public static MultiLayerNetwork pretrainedModelLeNet() {
		// 1) Load pretrained LeNet (MNIST 10-class)
		ZooModel zoo = LeNet.builder()
				.numClasses(10) // MNIST pretrained weights are for 10 classes
				.build();

		MultiLayerNetwork pretrained;
		try {
			pretrained = (MultiLayerNetwork) zoo.initPretrained(PretrainedType.MNIST);
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed to load pretrained LeNet MNIST", e);
		}

		return pretrained;
	}
	
	// ======================================================================================================================

	// public static MultiLayerNetwork createMNIST_CNN_PretrainedLeNet(int workerId) {
	// 	// 1) Load pretrained LeNet (MNIST 10-class)
	// 	ZooModel zoo = LeNet.builder()
	// 			.numClasses(10) // MNIST pretrained weights are for 10 classes
	// 			.build();

	// 	MultiLayerNetwork pretrained;
	// 	try {
	// 		pretrained = (MultiLayerNetwork) zoo.initPretrained(PretrainedType.MNIST);
	// 	} catch (Exception e) {
	// 		throw new RuntimeException("Failed to load pretrained LeNet MNIST", e);
	// 	}

	// 	// 2) Replace the output layer for YOUR NUM_CLASSES (MNIST4 or MNIST10)
	// 	//    This keeps the conv feature extractor from the pretrained model,
	// 	//    but resets the classifier head.
	// 	FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
	// 			// optimizer not used by you (PSO), but required by the builder
	// 			.build();

	// 	String outputLayerName = pretrained.getLayerWiseConfigurations()
	// 			.getConf(pretrained.getnLayers() - 1)
	// 			.getLayer().getLayerName();

	// 	// If layer names are null (common), DL4J uses internal names; easiest:
	// 	// remove 1 layer from output and add a new output layer.
	// 	MultiLayerNetwork tl = new TransferLearning.Builder(pretrained)
	// 			.fineTuneConfiguration(ftc)
	// 			.removeLayersFromOutput(1)
	// 			.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
	// 					.nOut(NUM_CLASSES)              // <-- your config value (4 or 10)
	// 					.activation(Activation.SOFTMAX)
	// 					.build())
	// 			.build();

	// 	// 3) Seed is irrelevant now (weights loaded), but you can keep reproducibility elsewhere.
	// 	return tl;
	// }


	// public static MultiLayerNetwork createMNIST_CNN_PretrainedLeNet(int workerId) {

	// 	ZooModel zoo = LeNet.builder().numClasses(10).build();

	// 	MultiLayerNetwork pretrained;
	// 	try {
	// 		pretrained = (MultiLayerNetwork) zoo.initPretrained(PretrainedType.MNIST);
	// 	} catch (Exception e) {
	// 		throw new RuntimeException("Failed to load pretrained LeNet MNIST", e);
	// 	}

	// 	// Remove the last Dense(500->10) layer and replace it with Dense/Output(500->NUM_CLASSES)
	// 	MultiLayerNetwork tl = new TransferLearning.Builder(pretrained)
	// 			.removeLayersFromOutput(1)   // removes dense_2
	// 			.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
	// 					.nIn(500)            // <-- MUST be 500 (from summary)
	// 					.nOut(NUM_CLASSES)   // 4 for MNIST4
	// 					.activation(Activation.SOFTMAX)
	// 					.build())
	// 			.build();

	// 	return tl;
	// }

	public static MultiLayerNetwork createMNIST_CNN_PretrainedLeNet(int workerId) {

		// Pretrained Model ===========================================================
		ZooModel zoo = LeNet.builder().numClasses(10).build();

		MultiLayerNetwork pretrained;
		try {
			pretrained = (MultiLayerNetwork) zoo.initPretrained(PretrainedType.MNIST);
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed to load pretrained LeNet MNIST", e);
		}

		// ============================================================================
		// DL4J needs a FineTuneConfiguration to define the updater (Adam, SGD, learning rate )
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.updater(new NoOp())   // <-- prevents optimizer assumptions
				.build();

		// From your summary: last classifier layer had nIn=500
		MultiLayerNetwork tl = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)     // <-- REQUIRED in 1.0.0-M2.1
				.removeLayersFromOutput(2)      // remove 2 layers
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(500)
						.nOut(NUM_CLASSES)     // 4 or 10 depending on your cfg
						.activation(Activation.SOFTMAX)	// OutputLayer in DL4J contains its own activation function (softmax / sigmoid / etc.)	
														// this depends on the methodology used to define activation layers. They can be embedded or
														// be external (right afterwards) to dense layers
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();

		return tl;
	}

	// ======================================================================================================================
	// Iris Dataset Model Architecture 

	public static MultiLayerNetwork createIrisModel(int workerId) {
		if(printModel) {
			System.out.println("Using Iris Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId) // or pass seed from outside
				.weightInit(WeightInit.XAVIER)				
				.list()
				.layer(new DenseLayer.Builder() // Hidden Layer 1 (with input Layer)
						.nIn(NUM_FEATURES)
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
						.lossFunction(LossFunctions.LossFunction.MCXENT)  // is used only for model.fit(...). Ignore it, we arent doing backpropagation
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

	public static MultiLayerNetwork createWineModel(int workerId) {
		if(printModel) {
			System.out.println("Using Wine Model");
		}

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(NUM_FEATURES)
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
        // For Hidden Layer 1   => 13 * 32 + 32 
        // For Hidden Layer 2   => 32 * 16 + 16
        // For Output Layer     => 16  * 3 + 3
        // 1027 weights 

	// ======================================================================================================================
	// MNIST Dataset Model Architecture 

	public static MultiLayerNetwork createMNISTModelMLP(int workerId) {
		if(printModel) {
			System.out.println("Using MNIST Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(128)
						.nOut(64)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(64)
						.nOut(NEURAL_OUTPUT)
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// Number of weights in the network calculation:  
        // For Hidden Layer 1   => 784 * 256 + 256  
        // For Hidden Layer 2   => 256 * 128 + 128
        // For Output Layer     => 128 * 10 + 10
        // 235146 weights all in all
		// 400000 == NN400K
		// this is comparable to NN400K

		// 1048576
		// 1881444
		//  940584

	// ======================================================================================================================

	public static MultiLayerNetwork createMNISTModelMLPSimple_0(int workerId) {
		if(printModel) {
			System.out.println("Using MNIST Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)
						.nOut(64)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(64)
						.nOut(32)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(32)
						.nOut(NEURAL_OUTPUT)
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================

	public static MultiLayerNetwork createMNISTModelMLPSimple_1(int workerId) {
		if (printModel) {
			System.out.println("Using MNIST Tiny MLP (1 hidden layer, PSO-friendly)");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(NUM_FEATURES)
						.nOut(32)
						.activation(Activation.TANH) // smooth for PSO, like you used elsewhere
						.build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(32)
						.nOut(NEURAL_OUTPUT)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.feedForward(NUM_FEATURES))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================

	public static MultiLayerNetwork createMNISTModelMLPSimple_2(int workerId) {
		if (printModel) {
			System.out.println("Using MNIST Ultra-Simple Model (no hidden / logistic regression)");
		}

		int nIn = NUM_FEATURES;       // should be 28*28 = 784
		int nOut = NEURAL_OUTPUT;     // should be 10

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(nIn)
						.nOut(nOut)
						.activation(Activation.SOFTMAX)
						.build())
				// Helps DL4J infer shapes cleanly for a single-layer net:
				.setInputType(InputType.feedForward(nIn))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}	// 784 * 10 + 10 = 7850 parameters	 

	// ======================================================================================================================

    public static MultiLayerNetwork createMNIST4MLP(int workerId) {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123 + workerId)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(NUM_FEATURES)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
                        .nIn(64)
                        .nOut(NEURAL_OUTPUT)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }
	// Weight Calculation: 
	// 784×128 + 128 = 100,480
	// 128×64 + 64 = 8,256
	// 64×4 + 4 = 260
	// total = 100,480 + 8,256 + 260 = 108,996 weights

	// ======================================================================================================================

	public static MultiLayerNetwork createMNISTCnn_New_2(int workerId) {

		if (printModel) {
			System.out.println("Using MNIST4 CNN (PSO-feasible)");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)	// 28 x 28 x 1
						.nIn(1)
						.nOut(8)       
						.stride(1, 1)
						.padding(0, 0)							// 26 x 26 x 8
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// 13 x 13 x 8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)				// 11 x 11 x 16
						.nOut(16)     
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// 5 x 5 x 16
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder()
						.nOut(32)                // keep 32 (good PSO control knob)
						.activation(Activation.TANH)  // smoother than ReLU for PSO
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)	
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())

				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================

	public static MultiLayerNetwork createMNISTModelCNNHeavy(int workerId) {

		int height = 28, width = 28, channels = 1;
		int nOut = 10;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(0, new ConvolutionLayer.Builder(3, 3)
						.nIn(channels)
						.stride(1, 1)
						.padding(0, 0)
						.nOut(32)
						.activation(Activation.RELU)
						.build())

				// MaxPooling2D(pool=2x2, stride=2, valid)
				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.padding(0, 0)
						.build())

				// Conv2D(filters=64, kernel=3x3, stride=1, padding=valid=0, relu)
				.layer(2, new ConvolutionLayer.Builder(3, 3)
						.stride(1, 1)
						.padding(0, 0)
						.nOut(64)
						.activation(Activation.RELU)
						.build())

				// MaxPooling2D(pool=2x2, stride=2, valid)
				.layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.padding(0, 0)
						.build())
				.layer(5, new DenseLayer.Builder()
						.nOut(128)
						.activation(Activation.SIGMOID)
						.build())

				// Output(10, softmax, MCXENT)
				.layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nOut(nOut)
						.activation(Activation.SOFTMAX)
						.build())

				// If you feed flattened 784 vectors, DL4J will reshape to 1x28x28
				.setInputType(InputType.convolutionalFlat(height, width, channels))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================

	public static MultiLayerNetwork createMNIST4MLP_Reduced(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)   // 784
						.nOut(64)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(64)
						.nOut(32)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(32)
						.nOut(NEURAL_OUTPUT)  // 4
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// 784→64: 784×64 + 64 = 50,176 + 64 = 50,240
	// 64→32: 64×32 + 32 = 2,048 + 32 = 2,080
	// 32→4: 32×4 + 4 = 128 + 4 = 132
	// Total = 50,240 + 2,080 + 132 = 52,452 parameters

	// ======================================================================================================================

    public static MultiLayerNetwork createMNISTCnn(int workerId) {
		if(printModel) {
			System.out.println("Using CNN MNIST Model");
		}

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123 + workerId)
                .weightInit(WeightInit.RELU)
                .updater(new Adam(1e-3))
                .list()
                // Conv2D(16, 3, padding="same", use_bias=False)
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(16)		// Number of filters / feature maps == 16, output channels
                        .stride(1, 1)
                        .padding(1, 1)       // "same" for 3x3 with stride 1
                        .hasBias(false)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new ActivationLayer.Builder() 	// ReLU
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// MaxPooling2D()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)			// 3x3x32
                        .nOut(32)
                        .stride(1, 1)
                        .padding(1, 1)
                        .hasBias(false)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new ActivationLayer.Builder()		// ReLU
                        .activation(Activation.RELU)
                        .build())
                .layer(new GlobalPoolingLayer.Builder()		// GlobalAveragePooling2D() => Outputs 32 one for each channel / feature map
                        .poolingType(PoolingType.AVG)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)	// (num_classes, softmax)
                        .nOut(NUM_CLASSES)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(28, 28, 1)) // this is the input shape: h,w,c => (28,28,1)
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

		// 3 × 3 × 1 × 16 = 144 (Conv1)
		// 4 × 16 = 64 (Batch Norm) // 4 vectors per channel (feature maps), gamma, beta, mean, log10stdev 
		// 3 × 3 × 16 × 32 = 4608 (Conv2)
		// 4 × 32 = 128 (Batch Norm) 
		// 32 × 10 = 330 

		// Total = 144 + 64 + 4608 + 128 + 330 = 5274

		// As we can see the input size (28 x 28 = 784) is not included in these calculations. 
		// It determines though the number of convolution operations because thee already sized filter slides across the image 

		// For MACs input size plays a role: 
		// Even if this seems a small number of parameters / weights, it is much more computationally expensive to apply a forward pass to a CNN 
		// Rather than a Dense NN. MAC = Multiply–Accumulate (a sum)
			// weights are reused multiple times in forward pass we are convoluting.
			// In an MLP, 784 features connect directly to neurons once.
			// In a CNN, those 784 pixels are processed repeatedly via sliding kernels.
			// Conv1: MACs ≈ 28 × 28 × 16 × 9 = 112,896 MACs (3 X 3 = 9)
			// Conv2: MACs ≈ 14 × 14 × 32 × 144 = 903,168 MACs (3 X 3 X 16 = 144, since we have more )

        return model;
    }

	// ======================================================================================================================
	// MNIST4CNN


	public static MultiLayerNetwork createMNIST4Cnn(int workerId) {
		if (printModel) {
				System.out.println("Using CNN MNIST4 Model");
			}

			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.seed(123 + workerId)
					.weightInit(WeightInit.RELU)
					.list()
					.layer(new ConvolutionLayer.Builder(3, 3)	// 3 × 3 × 1 (because GrayScale) × 8 + 8 (Biases) = 80
							.nOut(8)
							.stride(1, 1)	// this is the convolutional step 
							.padding(1, 1)	// this is a padding of the input image so that output image gets same size
							.hasBias(true)           
							.activation(Activation.IDENTITY)  
							.build())
					.layer(new ActivationLayer.Builder()
							.activation(Activation.LEAKYRELU)	// An activation layer doesnt have parameters
							.build())
					.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)		// MaxPooling2D(2x2) => Sampling Layer => doesnt have parameters
							.kernelSize(2, 2)
							.stride(2, 2)
							.build())
					.layer(new ConvolutionLayer.Builder(3, 3)			//  3 * 3 * 8 * 16 + 16 (Biases) = 1168
							.nOut(16)
							.stride(1, 1)
							.padding(1, 1)
							.hasBias(true)
							.activation(Activation.IDENTITY)
							.build())
					.layer(new ActivationLayer.Builder()
							.activation(Activation.LEAKYRELU)
							.build())
					.layer(new GlobalPoolingLayer.Builder()		// GlobalAveragePooling2D()
							.poolingType(PoolingType.AVG)
							.build())
					.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)	// since not using model.fit(...), this loss function will never be used
							.nOut(NUM_CLASSES)                   // MNIST4 => 4 Classes 	16 × 4 + 4 = 68, // this project only uses model.output(...)
							.activation(Activation.SOFTMAX)
							.build())
					.setInputType(InputType.convolutional(28, 28, 1))
					.build();

			MultiLayerNetwork model = new MultiLayerNetwork(conf);
			model.init();
			return model;
	}
	
	// Total = 80 + 1168 + 68 = 1316

	// ======================================================================================================================

	public static MultiLayerNetwork createMNIST4Cnn_Simple(int workerId) {
		if (printModel) {
			System.out.println("Using MNIST4 CNN SIMPLE");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.RELU)  
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)
						.nOut(4)
						.stride(1, 1)
						.padding(1, 1)        
						.hasBias(true)
						.activation(Activation.IDENTITY)
						.build())
				.layer(new ActivationLayer.Builder()
						.activation(Activation.LEAKYRELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new GlobalPoolingLayer.Builder()
						.poolingType(PoolingType.AVG)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(4)
						.nOut(NUM_CLASSES)     
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================

	public static MultiLayerNetwork createMNIST4Cnn_New(int workerId) {

		if (printModel) {
			System.out.println("Using MNIST4 CNN");
		}

		int numClasses = NUM_CLASSES;   // MNIST4 => 4, MNIST => 10

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER) 
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)
						.nIn(1)
						.nOut(32)
						.stride(1, 1)
						.padding(0, 0)        
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)
						.nOut(64)
						.stride(1, 1)
						.padding(0, 0)           
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)
						.nOut(64)
						.stride(1, 1)
						.padding(0, 0)        
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nOut(32)              
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(numClasses)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))	// MNIST input: [batch, 1, 28, 28]
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// Params => k * k = 3 * 3 = 9, conv = nOut*(k*k*nIn + bias)
	// conv1: 32*(9*1+1)=320	// nIn = 1 because 1 channel because grayscale
	// conv2: 64*(9*32+1)=18496
	// conv3: 64*(9*64+1)=36928

	// Flatten means flattening the feature maps => 64 (number of channels) * 3 * 3 (dimensionality of the feature maps) = 576
	// dense: 576 * 64 + 64 = 36928
	// total 320 + 18496 + 36928 + 36928 = 92932 params
	// Reported Dimensionality: 92932

	public static MultiLayerNetwork createMNIST4Cnn_New_Simpler(int workerId) {

		if (printModel) {
			System.out.println("Using MNIST4 CNN");
		}

		int numClasses = NUM_CLASSES;   // MNIST4 => 4, MNIST => 10

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER) 
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)
						.nIn(1)
						.nOut(32)
						.stride(1, 1)
						.padding(0, 0)        
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)
						.nOut(64)
						.stride(1, 1)
						.padding(0, 0)           
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder()
						.nOut(32)           	  
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(numClasses)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================

	public static MultiLayerNetwork createMNIST4Cnn_New_2(int workerId) {

		if (printModel) {
			System.out.println("Using MNIST4 CNN (PSO-feasible)");
		}

		int numClasses = NUM_CLASSES;   // MNIST4 => 4, MNIST => 10

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)	// 28 x 28 x 1
						.nIn(1)
						.nOut(8)       
						.stride(1, 1)
						.padding(0, 0)							// 26 x 26 x 8
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// 13 x 13 x 8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)				// 11 x 11 x 16
						.nOut(16)     
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// 5 x 5 x 16
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder()
						.nOut(32)                // keep 32 (good PSO control knob)
						.activation(Activation.TANH)  // smoother than ReLU for PSO
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)	
						.nOut(numClasses)
						.activation(Activation.SOFTMAX)
						.build())

				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}
	// Input Layer is always considered: 
	// 9 * 1 (input) * 8 (output) + 8 = 80
	// 9 * 8 (input) * 16 + 16 = 1168
	// 32 * 5 x 5 x 16 + 32= 12832
	// 32 * 4 + 4 = 132
	// 80 + 1168 + 12832 + 132 = 14212 trainable parameters
	// ======================================================================================================================
	// SUSY Dataset Model Architecture 

	public static MultiLayerNetwork createSUSYModel_SOFTMAX(int workerId) {
		if(printModel) {
			System.out.println("Using SUSY Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)  // 18
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(128)
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(128)
						.nOut(NEURAL_OUTPUT)  // 2
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// Number of weights in the network calculation:  
		// For Hidden Layer 1   => 18 * 128 + 128  
		// For Hidden Layer 2   => 128 * 128 + 128
		// For Output Layer     => 128 * 2 + 2
		// 19202 weights all in all
		// 38018

	
	// ======================================================================================================================
	// SUSY Dataset Model Architecture - Binary Cross Entropy Loss

	public static MultiLayerNetwork createSUSYModel(int workerId) {
		if(printModel) {
			System.out.println("Using SUSY Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)  // 18
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(128)
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT) // binary cross-entropy
						.nIn(128)
						.nOut(NEURAL_OUTPUT)  
						.activation(Activation.SIGMOID)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================
	// Bank Dataset Model Architecture 

	public static MultiLayerNetwork createBankModel(int workerId) {
		int outputSize = 1;
		if(printModel) {
			System.out.println("Using BANK Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.updater(new Adam(1e-3))
				.l2(1e-4)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)
						.nOut(64)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(64)
						.nOut(64)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT) // binary cross-entropy
						.nIn(64)
						.nOut(NEURAL_OUTPUT)
						.activation(Activation.SIGMOID)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}
	
	// Number of weights in the network calculation:  
        // For Hidden Layer 1   => 53 * 64 + 64  
        // For Hidden Layer 2   => 64 * 64 + 64
        // For Output Layer     => 64 * 1 + 1
        // 7681 weights all in all
		// this is comparable to NN4K

	// ======================================================================================================================

	public static MultiLayerNetwork createBankModel40K(int workerId) {
		int outputSize = 1; 
		if(printModel) {
			System.out.println("Using BANK Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.updater(new Adam(1e-3))
				.l2(1e-4)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)
						.nOut(256)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(256)
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT) // binary cross-entropy
						.nIn(128)
						.nOut(outputSize)
						.activation(Activation.SIGMOID)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}
	
	// Number of weights in the network calculation:  
        // For Hidden Layer 1   => 53 * 256 + 256  
        // For Hidden Layer 2   => 256 * 128 + 128
        // For Output Layer     => 128 * 1 + 1
        // 46849 weights all in all
		// this is comparable to NN40K

	// ======================================================================================================================

	public static MultiLayerNetwork createAdultModel(int workerId) {

		if(printModel) {
			System.out.println("Using ADULT_INCOME Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.updater(new Adam(1e-3))
				.l2(1e-4)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)      // Adult input features after one-hot + scaling
						.nOut(64)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(64)
						.nOut(64)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT) // binary cross-entropy
						.nIn(64)
						.nOut(NEURAL_OUTPUT)
						.activation(Activation.SIGMOID)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// Number of weights in the network calculation:
	// For Hidden Layer 1   => 96 * 64 + 64
	// For Hidden Layer 2   => 64 * 64 + 64
	// For Output Layer     => 64 * 1 + 1
	// Total weights = 10433
	// this is comparable to NN40K)

	// ======================================================================================================================
	// COVERTYPE Dataset Model Architecture

	public static MultiLayerNetwork createCovertypeModel(int workerId) {
		if(printModel) {
			System.out.println("Using Covertype Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)    
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(128)
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(128)
						.nOut(NEURAL_OUTPUT)  
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// Number of weights in the network calculation:
	// For Hidden Layer 1   => 54 * 128 + 128
	// For Hidden Layer 2   => 128 * 128 + 128
	// For Output Layer     => 128 * 7 + 7
	// Total weights        => (54*128+128) + (128*128+128) + (128*7+7)
	//                      => (6912+128) + (16384+128) + (896+7)
	//                      => 7040 + 16512 + 903
	//                      => 24455 weights 
	// this is comparable to NN40K

	// ======================================================================================================================
	// HAR (UCI Human Activity Recognition) Dataset Model Architecture

	public static MultiLayerNetwork createHarModel(int workerId) {
		if(printModel) {
			System.out.println("Using HAR Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)  // 561
						.nOut(32)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(32)
						.nOut(32)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(32)
						.nOut(NEURAL_OUTPUT)  // 6
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}


	// Number of weights in the network calculation:
	// For Hidden Layer 1   => 561 * 32 + 32
	// For Hidden Layer 2   => 32 * 32 + 32
	// For Output Layer     => 32 * 6 + 6
	// 19046 weights all in all
	//
	// Breakdown:
	// Hidden1: 561*32 = 17952, +32 biases  = 17984
	// Hidden2: 32*32  = 1024,  +32 biases  = 1056
	// Output : 32*6   = 192,   +6 biases   = 198
	// Total  : 17984 + 1056 + 198 = 19038  <-- wait, check below
	//
	// Total is:
	// 17984 + 1056 + 198 = 19238

	// ======================================================================================================================
	// PENDIGITS Dataset Model Architecture

	public static MultiLayerNetwork createPendigitsModel(int workerId) {
		if(printModel) {
			System.out.println("Using PenDigits Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)	// .weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)   // 16 
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(128)
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(128)
						.nOut(NEURAL_OUTPUT) // 10 (digits 0..9)
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}


	// Number of weights in the network calculation:
	// For Hidden Layer 1   => 16 * 128 + 128
	// For Hidden Layer 2   => 128 * 128 + 128
	// For Output Layer     => 128 * 10 + 10
	// 19978 weights 

	// or with pendigits-half 

	// For Hidden Layer 1   => 16 * 128 + 128
	// For Hidden Layer 2   => 128 * 128 + 128
	// For Output Layer     => 128 * 5 + 5
	// 19333 weights 

	// ======================================================================================================================

	public static MultiLayerNetwork createPendigitsModelTanh(int workerId) {
		if(printModel) {
			System.out.println("Using PenDigits Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				// .weightInit(new UniformDistribution(-WEIGHTS_INIT_SCALE, WEIGHTS_INIT_SCALE)) 
				.weightInit(WeightInit.XAVIER)	// .weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)   // 16 
						.nOut(128)
						.activation(Activation.TANH)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(128)
						.nOut(128)
						.activation(Activation.TANH)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(128)
						.nOut(NEURAL_OUTPUT) // 10 (digits 0..9)
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		// model.params().muli(WEIGHTS_INIT_SCALE);
		return model;
	}

	// ======================================================================================================================

	public static MultiLayerNetwork createPendigitsModelSmaller(int workerId) {
		if (printModel) System.out.println("Using PenDigits PSO-friendly Model (TANH)");

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				// 16 -> 64
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)     // 16
						.nOut(64)
						.activation(Activation.TANH)
						.build())
				// 64 -> 64
				.layer(new DenseLayer.Builder()
						.nIn(64)
						.nOut(64)
						.activation(Activation.TANH)
						.build())
				// 64 -> 10
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(64)
						.nOut(NEURAL_OUTPUT)   // 10
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================

	public static MultiLayerNetwork createPendigitsModelSmaller_2(int workerId) {	// single hidden layer
		if (printModel) System.out.println("Using PenDigits PSO-friendly Model (TANH, small)");

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)
						.nOut(32)
						.activation(Activation.TANH)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(32)
						.nOut(NEURAL_OUTPUT)
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================

	public static MultiLayerNetwork createPendigitsModelSmaller_3(int workerId) {	// no hidden layer just weights connecting input and output layer ...
		if (printModel) System.out.println("Using PenDigits Ultra-Simple Model (no hidden)");

		int nIn = 16;           
		int nOut = NEURAL_OUTPUT;    

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(nIn)
						.nOut(nOut)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.feedForward(nIn)) 
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================
	// WineQuality Dataset Model Architecture

	public static MultiLayerNetwork createWineQualityModel(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				// Hidden Layer 1: nIn = inputDim, nOut = 12, relu
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)     // 12 in this case
						.nOut(12)
						.activation(Activation.RELU)
						.build())
				// Hidden Layer 2: nIn = 12, nOut = 9, relu
				.layer(new DenseLayer.Builder()
						.nIn(12)
						.nOut(9)
						.activation(Activation.RELU)
						.build())
				// Output Layer: nIn = 9, nOut = 1, sigmoid, binary cross-entropy
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
						.nIn(9)
						.nOut(NEURAL_OUTPUT)
						.activation(Activation.SIGMOID)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}


	// Number of weights in the network calculation:
	// Hidden Layer 1  => inputDim * 12 + 12
	// Hidden Layer 2  => 12 * 9 + 9
	// Output Layer    => 9 * 1 + 1
	//
	// Total params    => (inputDim * 12 + 12) + (12 * 9 + 9) + (9 * 1 + 1)
	//
	// If inputDim = 12:
	// Hidden Layer 1  => 12*12 + 12 = 156
	// Hidden Layer 2  => 12*9  + 9  = 117
	// Output Layer    => 9*1   + 1  = 10
	// total           => 156 + 117 + 10 = 283 weights

	// ======================================================================================================================
	// Letter

	public static MultiLayerNetwork createLetterModel(int workerId) {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)
						.nOut(128)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(128)
						.nOut(64)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder()
						.nIn(64)
						.nOut(NEURAL_OUTPUT)
						.lossFunction(LossFunctions.LossFunction.MCXENT) // softmax cross-entropy
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// Layer 1: 16 → 256
	// Weights: 16 * 256 = 4096
	// Biases: 256
	// Total: 4352
	// Layer 2: 256 → 256
	// Weights: 256 * 256 = 65536
	// Biases: 256
	// Total: 65792
	// Output: 256 → 26
	// Weights: 256 * 26 = 6656
	// Biases: 26
	// Total: 6682
	// Grand total
	// 4352 + 65792 + 6682 = 76826 parameters

	// ======================================================================================================================
	// Letter 70k (bigger)

	public static MultiLayerNetwork createLetterModel70K(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)
						.nOut(256)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(256)
						.nOut(256)
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder()
						.nIn(256)
						.nOut(NEURAL_OUTPUT)
						.lossFunction(LossFunctions.LossFunction.MCXENT) // softmax cross-entropy
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================
	// CIFAR3

    public static MultiLayerNetwork createCifar3Model(int workerId) {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123 + workerId)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(0, new ConvolutionLayer.Builder(3, 3)	// Conv(8, 3x3, same) + ReLU
                        .nIn(3)               // RGB input channels => thats why 3
                        .nOut(8)
                        .stride(1, 1)
                        .padding(1, 1)        
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nIn(8)
                        .nOut(16)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.AVG)
                        .build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(16)           
                        .nOut(NUM_CLASSES)      	// 3 for CIFAR-3
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(32, 32, 3))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

	// ======================================================================================================================

	public static MultiLayerNetwork createCifar3Model_PSO_Simple(int workerId) {

		if (printModel) {
			System.out.println("Using CIFAR3 CNN SIMPLE (PSO): Conv8 -> LeakyReLU -> Pool -> GAP -> Softmax");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.RELU)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)
						.nIn(3)                    // RGB
						.nOut(8)
						.stride(1, 1)
						.padding(1, 1)
						.hasBias(true)
						.activation(Activation.IDENTITY)
						.build())
				.layer(new ActivationLayer.Builder()
						.activation(Activation.LEAKYRELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new GlobalPoolingLayer.Builder()
						.poolingType(PoolingType.AVG)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(8)
						.nOut(NUM_CLASSES) 
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(32, 32, 3))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}


	// ======================================================================================================================

	public static MultiLayerNetwork createCifar3Model_New(int workerId) {	// Recommended

		if (printModel) {
			System.out.println("Using CIFAR3 CNN (Keras-style better): 32/64/64 -> Dense(64 relu) -> Softmax(3)");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3) // 3 * 3 * 3 * 32 = 864
						.nIn(3)
						.nOut(32)
						.stride(1, 1)
						.padding(0, 0)	// no padding this means input (32x32) => output (30x30)
						.activation(Activation.RELU)
						.build())

				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// input  (30x30) =>  output (15x15)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())

				.layer(new ConvolutionLayer.Builder(3, 3)	// 3 * 3 * 32 * 64 = 18432
						.nOut(64)										// input  (15x15) =>  output (13x13)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())

				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// input  (13x13) =>  output (6x6)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())

				.layer(new ConvolutionLayer.Builder(3, 3)	// 3 * 3 * 64 * 64 = 36864 
						.nOut(64)										// input  (6x6) =>  output (4x4)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()		// Flatten implicit (no need to explicitly define a Flatten Layer)
						.nOut(64)				// 4 * 4 * 64 = 1024 (flattend input), 1024 * 64 + 64 = 65600
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(32, 32, 3))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}
	// 896+18,496+36,928+65,600+195=122,115​
	// Recorded Dimensionality of the output is: 122115. 45% accuracy

	// ======================================================================================================================

	public static MultiLayerNetwork createCifar3Model_New_Simpler(int workerId) {

		if (printModel) {
			System.out.println("Using CIFAR3 CNN (Keras-style better): 32/64/64 -> Dense(64 relu) -> Softmax(3)");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()

				.layer(new ConvolutionLayer.Builder(3, 3) // 3 * 3 * 3 * 32 = 864
						.nIn(3)
						.nOut(32)
						.stride(1, 1)
						.padding(0, 0)	// no padding this means input (32x32) => output (30x30)
						.activation(Activation.RELU)
						.build())

				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// input  (30x30) =>  output (15x15)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())

				.layer(new ConvolutionLayer.Builder(3, 3)	// 3 * 3 * 32 * 64 = 18432
						.nOut(64)										// input  (15x15) =>  output (13x13)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())

				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// input  (13x13) =>  output (6x6)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder()		// Flatten implicit (no need to explicitly define a Flatten Layer)
						.nOut(64)				// 4 * 4 * 64 = 1024 (flattend input), 1024 * 64 + 64 = 65600
						.activation(Activation.RELU)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(32, 32, 3))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}
	// 896+18,496+36,928+65,600+195=122,115​
	// Recorded Dimensionality of the output is: 122115. 45% accuracy

	// ======================================================================================================================

	public static MultiLayerNetwork createCifar3Model_New_Simpler_2(int workerId) {

		if (printModel) {
			System.out.println("Using CIFAR3 SIMPLE A: Conv(32) -> MaxPool -> Dense(64 relu) -> Softmax(3)");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)
						.nIn(3)
						.nOut(32)
						.stride(1, 1)
						.padding(1, 1)                 // SAME padding keeps 32x32
						.activation(Activation.RELU)
						.build())

				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)                  // 32x32 -> 16x16
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)	// 3 * 3 * 32 * 64 = 18432
						.nOut(64)										// input  (15x15) =>  output (13x13)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// input  (13x13) =>  output (6x6)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build()) 	
				// Big dense block (good for PSO search space)
				.layer(new DenseLayer.Builder()
						.nOut(32)
						.activation(Activation.RELU)
						
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nOut(NUM_CLASSES)             // = 3
						.activation(Activation.SOFTMAX)
						.build())

				.setInputType(InputType.convolutional(32, 32, 3))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================

	public static MultiLayerNetwork createCifar3Model_New_Simpler_3(int workerId) {

		if (printModel) {
			System.out.println("Using CIFAR3 SIMPLE B: Conv(16)->Pool->Conv(32)->Pool->Dense(64)->Softmax(3)");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)
						.nIn(3)
						.nOut(16)
						.stride(1, 1)
						.padding(1, 1)                 // SAME: 32x32
						.activation(Activation.RELU)
						.build())

				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)                  // 32->16
						.build())

				.layer(new ConvolutionLayer.Builder(3, 3)
						.nOut(32)
						.stride(1, 1)
						.padding(1, 1)                 // SAME: 16x16
						.activation(Activation.RELU)
						.build())

				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)                  // 16->8
						.build())

				.layer(new DenseLayer.Builder()
						.nOut(64)
						.activation(Activation.RELU)
						.build())

				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())

				.setInputType(InputType.convolutional(32, 32, 3))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}

	// ======================================================================================================================

	public static MultiLayerNetwork createCifar3Model_New_Simpler_4(int workerId) {

		if (printModel) {
			System.out.println("Using MNIST4 CNN (PSO-feasible)");
		}

		int numClasses = NUM_CLASSES;   // MNIST4 => 4, MNIST => 10

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)	// 28 x 28 x 1
						.nIn(1)
						.nOut(8)       
						.stride(1, 1)
						.padding(0, 0)							// 26 x 26 x 8
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// 13 x 13 x 8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)				// 11 x 11 x 16
						.nOut(16)     
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)	// 5 x 5 x 16
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder()
						.nOut(32)                // keep 32 (good PSO control knob)
						.activation(Activation.TANH)  // smoother than ReLU for PSO
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)	
						.nOut(numClasses)
						.activation(Activation.SOFTMAX)
						.build())

				.setInputType(InputType.convolutional(32, 32, 3))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return model;
	}
}

package utils;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.Adam;

public class Dl4jModelFactory {
        
	private static final Config cfg = Config.getInstance();
	private static final String DATASET = cfg.DATASET;
    public static final int NUM_FEATURES = cfg.NUM_FEATURES;
    public static final int NUM_CLASSES = cfg.NUM_CLASSES;
    public static final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;

	public static final boolean printModel = false;

	public static MultiLayerNetwork createModel() {
		// System.out.println("DATASET: " + DATASET);

		if("iris".equals(DATASET)) {
			return createIrisModel();

		} else if ("wine".equals(DATASET)) {
			return createWineModel();

		} else if ("mnist".equals(DATASET)) {
			return createMNISTModel();
				
		} else if ("susy".equals(DATASET)) {
			// return createSUSYModel_SOFTMAX();
			return createSUSYModel();

		} else if ("bank".equals(DATASET)) {
			// return createBankModel();
			return createBankModel40K();

		} else if ("adult".equals(DATASET)) {
			return createAdultModel();

		} else if ("covertype".equals(DATASET)) {
			return createCovertypeModel();

		} else if ("har".equals(DATASET)) {
			return createHarModel();

		} else if ("pendigits".equals(DATASET)) {
			return createPenDigitsModel();

		} else if ("winequality".equals(DATASET)) {
			return createWineQualityModel();

		} else if ("letter".equals(DATASET)) {
			return createLetterModel();
			// return createLetterModel70K();
		} else {
            throw new IllegalArgumentException("Invalid DATASET: " + DATASET);
		}
	}

	// ======================================================================================================================
	// Iris Dataset Model Architecture 

	public static MultiLayerNetwork createIrisModel() {
		if(printModel) {
			System.out.println("Using Iris Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123) // or pass seed from outside
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
		if(printModel) {
			System.out.println("Using Wine Model");
		}

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
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
        // 1027 weights all in all

	// ======================================================================================================================
	// MNIST Dataset Model Architecture 

	public static MultiLayerNetwork createMNISTModel() {
		if(printModel) {
			System.out.println("Using MNIST Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
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
	// SUSY Dataset Model Architecture 

	public static MultiLayerNetwork createSUSYModel_SOFTMAX() {
		if(printModel) {
			System.out.println("Using SUSY Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
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

	public static MultiLayerNetwork createSUSYModel() {
		if(printModel) {
			System.out.println("Using SUSY Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
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

	public static MultiLayerNetwork createBankModel() {
		int outputSize = 1;
		if(printModel) {
			System.out.println("Using BANK Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
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

	public static MultiLayerNetwork createBankModel40K() {
		int outputSize = 1; 
		if(printModel) {
			System.out.println("Using BANK Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
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

	public static MultiLayerNetwork createAdultModel() {

		if(printModel) {
			System.out.println("Using ADULT_INCOME Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
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

	public static MultiLayerNetwork createCovertypeModel() {
		if(printModel) {
			System.out.println("Using Covertype Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
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
	//                      => 24455 weights all in all
	// this is comparable to NN40K

	// ======================================================================================================================
	// HAR (UCI Human Activity Recognition) Dataset Model Architecture

	public static MultiLayerNetwork createHarModel() {
		if(printModel) {
			System.out.println("Using HAR Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
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
	// NOTE: Correct total is:
	// 17984 + 1056 + 198 = 19238
	// (So the correct "all in all" number is 19238 parameters.)

	// ======================================================================================================================
	// PENDIGITS Dataset Model Architecture

	public static MultiLayerNetwork createPenDigitsModel() {
		if(printModel) {
			System.out.println("Using PenDigits Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
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

	// ======================================================================================================================
	// WineQuality Dataset Model Architecture

	public static MultiLayerNetwork createWineQualityModel() {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
				.list()
				// Hidden Layer 1: nIn=inputDim, nOut=12, relu
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)     // 12 in your case
						.nOut(12)
						.activation(Activation.RELU)
						.build())
				// Hidden Layer 2: nIn=12, nOut=9, relu
				.layer(new DenseLayer.Builder()
						.nIn(12)
						.nOut(9)
						.activation(Activation.RELU)
						.build())
				// Output Layer: nIn=9, nOut=1, sigmoid, binary cross-entropy
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


	// ======================================================================================================================
	// Number of weights in the network calculation (same style as your PenDigits comment):
	//
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
	// TOTAL           => 156 + 117 + 10 = 283 weights

	// ======================================================================================================================
	// Letter

	public static MultiLayerNetwork createLetterModel() {
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
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

	// 	Layer 1: 16 → 256
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


	public static MultiLayerNetwork createLetterModel70K() {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
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

}

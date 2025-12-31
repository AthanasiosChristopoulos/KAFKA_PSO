package utils;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.Adam;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;


public class Dl4jModelFactory {
        
	private static final Config cfg = Config.getInstance();
	private static final String DATASET = cfg.DATASET;
    public static final int NUM_FEATURES = cfg.NUM_FEATURES;
    public static final int NUM_CLASSES = cfg.NUM_CLASSES;
    public static final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;

	public static final boolean printModel = true;

	public static MultiLayerNetwork createModel() {
		// System.out.println("DATASET: " + DATASET);

		if("iris".equals(DATASET)) {
			return createIrisModel();

		} else if ("wine".equals(DATASET)) {
			return createWineModel();

		} else if ("mnist".equals(DATASET)) {
			// return createMNISTModel();
			return createMNISTCnn();
				
		} else if ("mnist4".equals(DATASET)) {
			// return createMNISTModel();
			return createMNIST4Cnn();
				
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

		} else if ("pendigits".equals(DATASET) || "pendigits-half".equals(DATASET)) {
			return createPendigitsModel();

		} else if ("winequality".equals(DATASET)) {
			return createWineQualityModel();

		} else if ("letter".equals(DATASET)) {
			return createLetterModel();
			// return createLetterModel70K();
		} else if ("cifar3".equals(DATASET)) {

			return createCifar3Model();
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

    public static MultiLayerNetwork createMNISTCnn() {
		if(printModel) {
			System.out.println("Using CNN MNIST Model");
		}
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.RELU)
                .updater(new Adam(1e-3))
                .list()
                // Conv2D(16, 3, padding="same", use_bias=False)
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(16)		// Number of filters / feature maps
                        .stride(1, 1)
                        .padding(1, 1)       // "same" for 3x3 with stride 1
                        .hasBias(false)
                        .activation(Activation.IDENTITY)
                        .build())
                // BatchNorm
                .layer(new BatchNormalization.Builder().build())
                // ReLU
                .layer(new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build())
                // MaxPooling2D()
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Conv2D(32, 3, padding="same", use_bias=False)
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(32)
                        .stride(1, 1)
                        .padding(1, 1)
                        .hasBias(false)
                        .activation(Activation.IDENTITY)
                        .build())
                // BatchNorm
                .layer(new BatchNormalization.Builder().build())
                // ReLU
                .layer(new ActivationLayer.Builder()
                        .activation(Activation.RELU)
                        .build())
                // GlobalAveragePooling2D()
                .layer(new GlobalPoolingLayer.Builder()		// Outputs 32 one for each channel / feature map
                        .poolingType(PoolingType.AVG)
                        .build())
                // Dense(num_classes, softmax) + sparse categorical crossentropy
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
                        .nOut(NUM_CLASSES)
                        .activation(Activation.SOFTMAX)
                        .build())
                // This is the key replacement for Keras Reshape((28,28,1))
                .setInputType(InputType.convolutional(28, 28, 1)) // h,w,c
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

		// 3 × 3 × 1 × 16 = 144 (Conv1)
		// 4 × 16 = 64 (Batch Norm) // 4 vectors per channel (feature maps), gamma, beta, mean, log10stdev 
		// 3 × 3 × 16 × 32 = 4608 (Conv2)
		// 4 × 32 = 128 (Batch Norm) 
		// 32 × 10 = 330 

		// Total = 144 + 64 + 4608 + 128 + 330 = 5274
		
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

	public static MultiLayerNetwork createMNIST4Cnn() {
		if (printModel) {
			System.out.println("Using CNN MNIST4 Model (8/16 filters)");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123)
				.weightInit(WeightInit.RELU)
				.updater(new Adam(1e-3))
				.list()

				// Conv2D(8, 3, padding="same", use_bias=False)
				.layer(new ConvolutionLayer.Builder(3, 3)
						.nOut(8)                 // 8 filters / feature maps
						.stride(1, 1)
						.padding(1, 1)           // "same" for 3x3 stride 1
						.hasBias(false)
						.activation(Activation.IDENTITY)
						.build())

				// BatchNorm
				.layer(new BatchNormalization.Builder().build())

				// ReLU
				.layer(new ActivationLayer.Builder()
						.activation(Activation.RELU)
						.build())

				// MaxPooling2D()
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())

				// Conv2D(16, 3, padding="same", use_bias=False)
				.layer(new ConvolutionLayer.Builder(3, 3)
						.nOut(16)                // 16 filters / feature maps
						.stride(1, 1)
						.padding(1, 1)
						.hasBias(false)
						.activation(Activation.IDENTITY)
						.build())

				// BatchNorm
				.layer(new BatchNormalization.Builder().build())

				// ReLU
				.layer(new ActivationLayer.Builder()
						.activation(Activation.RELU)
						.build())

				// GlobalAveragePooling2D()
				.layer(new GlobalPoolingLayer.Builder()
						.poolingType(PoolingType.AVG)
						.build())

				// Dense(num_classes, softmax) + sparse categorical crossentropy
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)                 // for MNIST4 => 4
						.activation(Activation.SOFTMAX)
						.build())

				// Input: (H,W,C) = (28,28,1)
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		if (printModel) {
			System.out.println(model.summary());
			System.out.println("Trainable params: " + model.numParams());
		}

		return model;
	}

	// Parameter count (for sanity)

	// Conv1: 3×3×1×8 = 72

	// BN1: 4×8 = 32

	// Conv2: 3×3×8×16 = 1,152

	// BN2: 4×16 = 64

	// Output: 16×4 + 4 = 68
	// Total = 72 + 32 + 1152 + 64 + 68 = 1,388 params

	// So PSO dimension should be 1388 for this model.

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

	public static MultiLayerNetwork createPendigitsModel() {
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

	// or with pendigits-half 

	// For Hidden Layer 1   => 16 * 128 + 128
	// For Hidden Layer 2   => 128 * 128 + 128
	// For Output Layer     => 128 * 5 + 5
	// 19333 weights 

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

	// ======================================================================================================================
	// Letter 70k (bigger)

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

	// ======================================================================================================================
	// CIFAR3

    public static MultiLayerNetwork createCifar3Model() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()

                // Conv(8, 3x3, same) + ReLU
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(3)               // RGB channels
                        .nOut(8)
                        .stride(1, 1)
                        .padding(1, 1)        // "same" padding for 3x3 stride1
                        .activation(Activation.RELU)
                        .build())

                // MaxPool(2x2)
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())

                // Conv(16, 3x3, same) + ReLU
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nIn(8)
                        .nOut(16)
                        .stride(1, 1)
                        .padding(1, 1)
                        .activation(Activation.RELU)
                        .build())

                // MaxPool(2x2)
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())

                // GlobalAveragePooling
                .layer(4, new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.AVG)
                        .build())

                // Dense(numClasses) + Softmax (MCXENT)
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(16)              // after global avg pooling: one value per channel
                        .nOut(NUM_CLASSES)      // 3 for CIFAR-3
                        .activation(Activation.SOFTMAX)
                        .build())

                // Tell DL4J the input shape (height, width, channels) - These are images not separate featuresd
                .setInputType(InputType.convolutional(32, 32, 3))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

}

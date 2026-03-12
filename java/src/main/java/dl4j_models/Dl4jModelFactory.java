package dl4j_models;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import utils.Config;
import utils.NhwcToFeedForwardPreProcessor;

import java.io.File;

import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.learning.config.Adam;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;

import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.common.primitives.Pair;

public class Dl4jModelFactory {
        
	private static final Config cfg = Config.getInstance();
	private static final String DATASET = cfg.DATASET;
    public static final int NUM_FEATURES = cfg.NUM_FEATURES;
    public static final int NUM_CLASSES = cfg.NUM_CLASSES;
    public static final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;
    private static final float WEIGHTS_INIT_SCALE = cfg.WEIGHTS_INIT_SCALE;

	public static final boolean printModel = false;

	public static Pair<PsoModel, Integer> createModel(int workerId, boolean preTrained) {
		// System.out.println("DATASET: " + DATASET);
		PsoModel model = null;
		int head_layer_idx = -1;
		Pair<PsoModel, Integer> pair = null;

		if("iris".equals(DATASET)) {
			// model = createIrisModel(workerId);
			model = createDenseModel_1(workerId);
			// model = createDenseModel_2(workerId);
			// model = createDenseModel_3(workerId);

		} else if ("wine".equals(DATASET)) {
			model = createWineModel(workerId);

		} else if ("susy".equals(DATASET)) {
			// model = createSUSYModel_SOFTMAX(workerId);
			model = createSUSYModel(workerId);

		} else if ("bank".equals(DATASET)) {
			// model = createBankModel(workerId);
			model = createBankModel40K(workerId);

		} else if ("adult".equals(DATASET)) {
			model = createAdultModel(workerId);

		} else if ("covertype".equals(DATASET)) {
			model = createCovertypeModel(workerId);

		} else if ("har".equals(DATASET)) {
			model = createHarModel(workerId);

		} else if (DATASET.contains("pendigits")) {
			// model = createPendigitsModelTanh(workerId);
			model = createDenseModel_1(workerId);	// forward pass cost: CPU = 10ms / GPU = 3ms
			// model = createDenseModel_2(workerId);
			// model = createDenseModel_3(workerId);
			// model = createDenseModel_4(workerId);

		} else if ("winequality".equals(DATASET)) {
			// model = createWineQualityModel(workerId);
			// model = createDenseModel_1(workerId);
			model = createDenseModel_2(workerId);
			// model = createDenseModel_3(workerId);

		} else if ("letter".equals(DATASET)) {
			model = createLetterModel(workerId);
			// model = createLetterModel70K(workerId);

		} else if ("mnist5".equals(DATASET)) {	// Forward pass cost: CPU => 200ms / GPU => 30ms  
			// model = createMNISTModelMLP(workerId);
			// model = createMNISTModelMLPSimple_1(workerId);
			// model = createMNISTModelMLPSimple_2(workerId);
			// model = createMNIST5Cnn(workerId);	// 70ms forward pass
			// model = createMNIST5Cnn_Simple(workerId);	// 25ms forward pass on average
			// model = createMNIST5MLP(workerId);
			// model = createMNIST5MLP_Reduced(workerId);
			// model = createMNIST5Cnn_New(workerId);			// this costs on forward pass much more time (60ms)
			// model = createMNIST5Cnn_New_Simpler(workerId);
			// model = createMNIST5Cnn_New_2(workerId);		// 0.89
			// model = createMNIST5Cnn_New_3(workerId);		//
			// model = createMNIST5Cnn_New_4(workerId);		// 0.915
			// model = createMNIST5Cnn_New_4_without_2_Dense(workerId);	

			// model = createMNIST5Cnn_New_5(workerId);		// 0.385 with GlobalPooling Layer
			// model = createMNIST5Cnn_New_6(workerId);		// 86%
			// model = createMNIST5Cnn_New_7(workerId); 		// 0.37, with GlobalPooling Layer
			// model = createMNIST5Cnn_New_8(workerId); 		// 0.795
			// model = createMNIST5Cnn_New_9(workerId); 	// 92%
			// model = createMNIST5Cnn_New_10(workerId); 		// 0.935
			// model = createMNIST5Cnn_New_11(workerId); 		// 0.925
			// model = createMNIST5Cnn_New_12(workerId);		// 0.935
			// model = createDenseModel_1(workerId);
			// model = createMNIST5Cnn_New_13(workerId);		// Accuracy:0.92333335
			model = createMNIST5Cnn_New_14(workerId);		// Accuracy:0.933

	// ======================================================================================================================

		} else if ("mnist".equals(DATASET)) {

			// cfg.USING_PRETRAINED_MODEL = false;
			// model = createMNISTModelMLP(workerId);			
			// model = createMNISTModelMLPSimple_0(workerId);
			// model = createMNISTModelMLPSimple_1(workerId);
			// model = createMNISTModelMLPSimple_2(workerId);
			// model = createMNISTCnn(workerId);
			// model = createMNIST5Cnn_New(workerId);
			// model = createMNISTCnn_New_2(workerId);
			// model = createMNISTModelCNNHeavy(workerId);

			// model = createMNIST5Cnn_New_4_without_2_Dense(workerId); 	// 0.295
			// model = createMNIST5Cnn_New_9(workerId); 	// 62%
			// model = createMNIST5Cnn_New_10(workerId);	// 65%
			// model = createMNIST5Cnn_New_12(workerId);	// 0.6433333
			// model = createMNIST5Cnn_New_13(workerId); // 0.7366667
			// model = createMNIST5Cnn_New_14(workerId); // bestAccuracy: 0.69
			// model = createDenseModel_1(workerId);	// 0.6066667

			cfg.USING_PRETRAINED_MODEL = true;
			
			if(cfg.USING_PRETRAINED_MODEL) {
				
				int version = 11;

				String filename;
				switch (version) {
					case 1 -> filename = "../python/pretrained_model/mnist_base_plus_head_v1.h5";	
						// best, 99%
					case 2 -> filename = "../python/pretrained_model/mnist_base_plus_head_v2.h5";
						// 0.83
					case 3 -> filename = "../python/pretrained_model/mnist_base_plus_head_v3.h5";
					case 4 -> filename = "../python/pretrained_model/mnist_base_plus_head_v4.h5";
					case 5 -> filename = "../python/pretrained_model/mnist_base_plus_head_v5.h5";
					case 6 -> filename = "../python/pretrained_model/mnist_base_plus_head_v6.h5";
					case 7 -> filename = "../python/pretrained_model/mnist_base_plus_head_v7.h5";
					case 8 -> filename = "../python/pretrained_model/fmnist_base_plus_head_v3.h5";	
						// 58%	
					case 9 -> filename = "../python/pretrained_model/fmnist_base_plus_head_v2.h5";				
						// 70%
						// recomendation
					// case 10 -> filename = "../python/pretrained_model/svhn_flat_dense_v1_best.h5";					
					case 10 -> filename = "../python/pretrained_model/svhn_28x28x1_v2_mnist_best.h5";					
						// 0.756 accuracy after PSO training, 66.4% in the pretrained
					case 11 -> filename = "../python/pretrained_model/svhn_28x28x1_v4_mnist_final.h5";				
						// 0.82 PSO training, 0.652 pretrained (also 0.8466667)
						// even if pretty slow it will keep improving
						// recommended
					case 12 -> filename = "../python/pretrained_model/svhn_28x28x1_v3_mnist_final.h5";				
						// 0.785 PSO training, 0.676 pretrained
					case 13 -> filename = "../python/pretrained_model/fmnist_base_plus_head_v4.h5";
						// 0.693333 PSO training UnFreezeLvl = 0
					case 14 -> filename = "../python/pretrained_model/fmnist_base_plus_head_v5.h5";
						//  0.62333333,
					case 15 -> filename = "../python/pretrained_model/svhn_v5_mnist_final.h5";
						// 0.78
					case 16 -> filename = "../python/pretrained_model/svhn_v6_mnist_final.h5";
						// 0.74
					case 17 -> filename = "../python/pretrained_model/fmnist_base_plus_head_v1.h5";		
						// 61.6%
					case 18 -> filename = "../python/pretrained_model/fmnist_base_plus_head_v2_1.h5";		
						// 77%, recommended for fashion_mnist
					case 19 -> filename = "../python/pretrained_model/svhn_v7_mnist_final.h5";		
						// 78%	
					case 20 -> filename = "../python/pretrained_model/svhn_v8_mnist_final.h5";		
						// 81%
					case 21 -> filename = "../python/pretrained_model/fmnist_base_plus_head_v6.h5";		
						// 63%
					case 22 -> filename = "../python/pretrained_model/fmnist_base_plus_head_v7.h5";		
						// 35%
					case 23 -> filename = "../python/pretrained_model/fmnist_base_plus_head_v8.h5";		
						// 66%
					case 24 -> filename = "../python/pretrained_model/svhn_v9_mnist_final.h5";		
						// 78%
					case 25 -> filename = "../python/pretrained_model/svhn_v10_mnist_final.h5";	
					default -> filename = "no_pretrained_file_chosen";
				}
				
				if (preTrained) {
					System.out.println("Using model version: " + version);

					switch (version) {
						case -2, -1, 0 -> model = pretrainedModelLeNet();
						case 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
						24, 25 -> model = pretrainedModelMNIST(filename);
						default -> throw new IllegalArgumentException("Unknown version: " + version);
					}

				} else {

					switch (version) {
						case -2 -> pair = createMNIST_CNN_PretrainedLeNet_v1(workerId);		// 0.9
						case -1 -> pair = createMNIST_CNN_PretrainedLeNet_v2(workerId);
						case 0 -> pair = createMNIST_CNN_PretrainedLeNet_v3(workerId);

						case 1 -> pair = createCNNModel_1_Layer(workerId, filename, 64);	// 0.99, fine-tuneable 0.9
						case 2 -> pair = createCNNModel_1_Layer(workerId, filename, 800);
						case 3 -> pair = createCNNModel_1_Layer(workerId, filename, 128);		// 0.89
						case 4 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v4(workerId, filename, 50);
						case 5 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v5(workerId, filename, 128);	// 0.7
						case 6 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v6(workerId, filename, 256);	// 0.53
						// case 6 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v6_1(workerId, filename, 64);	// 0.71
						// case 7 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v7(workerId, filename);	// 0.77
						// case 7 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v7_1(workerId, filename, 64);	// 0.46
						// case 7 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v7_2(workerId, filename, 10); // 0.66	
						// case 7 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v7_3(workerId, filename, 5 * 5 * 10);	
						// case 7 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v7_4(workerId, filename);	// 0.77
						case 7 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v7_5(workerId, filename);	// 0.84, 0.86 with freeze index 1
						// case 7 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v7_5_1(workerId, filename);	// 0.84, 0.86 with freeze index 1
						// case 7 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v7_6(workerId, filename, 64);	// 0.23
						case 8 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v7_5(workerId, filename);	// 0.72% partially frozen, 0.7% fully frozen
						case 9 -> pair = createCNNModel_1_Layer(workerId, filename, 800); // 80% Partial Freeze
						case 10 -> pair = createCNNModel_1_Layer(workerId, filename, 576);	
						case 11, 13 -> pair = createCNNModel_1_Layer(workerId, filename, 90);
						case 12 -> pair = createCNNModel_1_Layer(workerId, filename, 198);
						case 14 -> pair = createCNNModel_1_Layer(workerId, filename, 200);
						case 15 -> pair = createCNNModel_1_Layer(workerId, filename, 800);
						case 16 -> pair = createCNNModel_1_Layer(workerId, filename, 128);
						case 17 -> pair = createMNIST_CNN_Pretrained_MNIST_Simpler_v1(workerId, filename, 800);	// 0.8, fine-tuneable 0.7
						case 18 -> pair = createCNNModel_1_Layer(workerId, filename, 400);
						case 19 -> pair = createCNNModel_1_Layer(workerId, filename, 784);
							// 78%
						case 20 -> pair = createCNNModel_1_Layer(workerId, filename, 144);
						case 21 -> pair = createCNNModel_1_Layer(workerId, filename, 36);
						case 22 -> pair = createCNNModel_1_Layer(workerId, filename, 27);
						case 23 -> pair = createCNNModel_1_Layer(workerId, filename, 98);
						case 24 -> pair = createCNNModel_1_Layer(workerId, filename, 98);
						case 25 -> pair = createCNNModel_1_Layer(workerId, filename, 98);
							// 81.3%

						default -> throw new IllegalArgumentException("Unknown version: " + version);
					}
				}
			}

		// ======================================================================================================================	

		} else if ("fashion_mnist".equals(DATASET)) {

			cfg.USING_PRETRAINED_MODEL = true;

			// model = createMNISTModelMLPSimple_2(workerId);
			// model = createMNISTModelMLPSimple_1(workerId);
			// model = createMNIST5Cnn_New_Simpler(workerId);
			// model = createMNISTCnn_New_2(workerId);
			// model = createMNISTCnn(workerId);
			// model = createMNIST5Cnn_New(workerId);
			
			// pretrained =============================================================================================
			
			if(cfg.USING_PRETRAINED_MODEL) {

				int version = 1;

				String filename;
				switch (version) {
					case 1 -> filename = "../python/pretrained_model/mnist_base_plus_head_v2.h5";	
					default -> filename = "no_pretrained_file_chosen";
				}
				
				if (preTrained) {
					System.out.println("Using model version: " + version);
					switch (version) {
						case 1 -> model = pretrainedModelMNIST(filename);
						default -> throw new IllegalArgumentException("Unknown version: " + version);
					}

				} else {
					switch (version) {
						case 1 -> pair = createCNNModel_1_Layer(workerId, filename, 800);
						default -> throw new IllegalArgumentException("Unknown version: " + version);
					}
				}
			}

		// ==============================================================================================

		} else if (DATASET.contains("cifar")) {

			// model = createCifar3Model_PSO_Simple(workerId);
			// model = createCifar3Model(workerId);
			// model = createCifar3Model_New(workerId);	
			// model = createCifar3Model_New_Simpler(workerId);
			// model = createCifar3Model_New_Simpler_2(workerId);
			// model = createCifar3Model_New_Simpler_3(workerId);
			// model = createCifar3Model_New_Simpler_4(workerId);
			// model = createMNIST5Cnn_New_Simpler(workerId);
			// model = buildCifarNCHW(3);
			// model = createCifarCnn_New_13(workerId);	// failure !!!

			cfg.USING_PRETRAINED_MODEL = true;

			// pretrained =============================================================================================

			if(cfg.USING_PRETRAINED_MODEL) {

				int version = 16;
				String filename;
				
				if(version == 4 || version == 5) {
					cfg.TRANSFORM_IMAGE = true;
					// cfg.TRANSFORM_IMAGE = false;
				}

				switch (version) {
					case 1 -> filename = "../python/pretrained_model/cifar10_base_plus_head_v4.h5";
					case 2 -> filename = "../python/pretrained_model/mobilenetv2_base_32x32.h5";
					case 3 -> filename = "../python/pretrained_model/cifar100_pretrained_base.h5";
					case 4 -> filename = "../python/pretrained_model/mobilenetv2_base_224x224.h5";
					case 5 -> filename = "../python/pretrained_model/mobilenet_base_224x224.h5";
					case 6 -> filename = "../python/pretrained_model/tinyimagenet200_pretrained_v2.h5";		// 50%
					case 7 -> filename = "../python/pretrained_model/cifar10_base_plus_head_v5.h5";		// 81% (new - 500) vs (75% - cifar 10)
						// on cifar10 => 65%, on cifar5 => 83%
						// on cifar10 => 75%, on cifar5 => 83%
					case 8 -> filename = "../python/pretrained_model/cifar10_base_plus_head_v6.h5";		// 81% new one under certain circustances => 500
																									// 72% => 100 and 77% pretrained
					case 9 -> filename = "../python/pretrained_model/cifar100_base_plus_head_v5.h5";			// 67%
					case 10 -> filename = "../python/pretrained_model/stl10_pretrained_base_plus_head_v1.h5";	// 60%
					case 11 -> filename = "../python/pretrained_model/stl10_pretrained_resnet20_v1.h5";	// 60%
					case 12 -> filename = "../python/pretrained_model/cifar10_base_plus_head_v5_1.h5";	
						// Report on preTrained Model: 0.832, with nSamples: 500, nCorrect: 416 loss: 0.48254818
					case 13 -> filename = "../python/pretrained_model/cifar5_base_plus_head_v5_56789.h5";	
						// 57%
					case 14 -> filename = "../python/pretrained_model/cifar5_base_plus_head_v4_56789.h5";	
						// 55% 
					case 15 -> filename = "../python/pretrained_model/cifar5_base_plus_head_v4_01489.h5";	
						// Failure 53% accuracy at the most
					case 16 -> filename = "../python/pretrained_model/cifar10_base_plus_head_v4_half.h5";	
						// 0.6066667 for cifar 10
						// cifar5_half => 81%
						// pretrained model, no training => 0.334
					case 17 -> filename = "../python/pretrained_model/cifar10_base_plus_head_v5_half.h5";	
						// cifar10_half => 60%, cifar5_half => 86%
						// Recommended
					case 18 -> filename = "../python/pretrained_model/cifar10_base_plus_head_v6_half.h5";	
						// cifar5_half => 0.74
					case 19 -> filename = "../python/pretrained_model/cifar10_base_plus_head_v7_half.h5";	
						// cifar5_half => 0.8
					default -> throw new IllegalArgumentException("Unknown CIFAR pretrained version: " + version);
				}

				if (preTrained) {
					System.out.println("Using model version: " + version);

					switch (version) {
						case 1, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19 -> model = pretrainedModelCIFAR(filename);
						case 2, 4, 5, 11 -> model = pretrainedModelMobileNetV2(filename);
						default -> throw new IllegalStateException("Unknown ???" );
					}

				} else {

					switch (version) {
						case 1, 3, 10, 14, 15, 16 -> pair = createCIFAR_CNN_Pretrained_CIFAR_Simpler_v1_v4(workerId, filename, 128);
						case 7, 13, 17 -> pair = createCIFAR_CNN_Pretrained_CIFAR_Simpler_v1_v4(workerId, filename, 64); 		// 73% cifar10, 91% cifar5
							// cifar 10 trained 75%, cifar 5 optimized 90%
						case 6 -> pair = createCIFAR_CNN_Pretrained_CIFAR_Simpler_v6(workerId, filename, 200);
						case 9 -> pair = createCIFAR_CNN_Pretrained_CIFAR_Simpler_v1_v4(workerId, filename, 64); 		// 60% cifar5 (pretrained 0.014)
						case 8, 18 -> pair = createCIFAR_CNN_Pretrained_CIFAR_Simpler_v1_v4(workerId, filename, 384); 
							// 77% accuracy pretrained, 75% new head
						case 12 -> pair = createCNN_1_L(workerId, filename, 64, 25); 

						case 2, 4 -> pair = createCifarFromMobileNetV2Base(workerId, filename); 
						case 11-> pair = createCifarFromResNet20Stl10(workerId, filename);
						case 5 -> pair = createCifarFromMobileNet(workerId, filename);
						case 19 -> pair = createCIFAR_CNN_Pretrained_CIFAR_Simpler_v1_v4(workerId, filename, 192);

						default -> throw new IllegalStateException("Unknown ???");
					}
				}
			}

		// ======================================================================================================================	

		} else if ("svhn".equals(DATASET)) {

			cfg.USING_PRETRAINED_MODEL = true;

			// pretrained =============================================================================================
			if(cfg.USING_PRETRAINED_MODEL) {
				
				int version = 1;

				String filename;
				switch (version) {
					case 1 -> filename = "../python/pretrained_model/svhn_v4_final.h5";		// NO FREEZE 69%, FULL freeze 81%, 80% Partial Freeze					
					default -> filename = "no_pretrained_file_chosen";
				}
				
				if (preTrained) {
					switch (version) {
						case 1 -> model = pretrainedModelMNIST(filename);
						default -> throw new IllegalArgumentException("Unknown version: " + version);
					}

				} else {

					switch (version) {
						case 1 -> pair = createCNNModel_1_Layer(workerId, filename, 96); // 80% Partial Freeze
							// 83) bestAccuracy: 0.51

						default -> throw new IllegalArgumentException("Unknown version: " + version);
					}
				}
			}

		// ======================================================================================================================	

		} else {
            throw new IllegalArgumentException("Invalid DATASET: " + DATASET);
		}

		if(cfg.FREEZE == false) {
			cfg.USING_PRETRAINED_MODEL = false;
		}

		if(pair == null) {
			return Pair.of(model, head_layer_idx);
		} else {
			return pair;
		}
	}

	// ============================================================================

	public static Pair<PsoModel, Integer> createCIFAR_CNN_Pretrained_CIFAR_Simpler_v3(
			int workerId, String fileName, int inputDim) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = pretrainedModelCIFAR(fileName).asMultiLayerNetwork();

		// ============================================================================
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(1)
			.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(64)      
						.nOut(NUM_CLASSES)     
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.biasInit(0.0)
						.build())
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ============================================================================

	public static Pair<PsoModel, Integer> createCIFAR_CNN_Pretrained_CIFAR_Simpler_v1_v4(int workerId, String fileName, int inputDim) {

		MultiLayerNetwork pretrained = pretrainedModelCIFAR(fileName).asMultiLayerNetwork();

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();

		int start = (int) new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(1 + cfg.FREEZE_INDEX)
			.build().numParams();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(1)
			.build();
		
		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)
				// .setFeatureExtractor(7)	// look at model.summary()
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(inputDim)           // for this TF model: 128
						.nOut(NUM_CLASSES) 
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.biasInit(0.0)
						.build())
				.build();

		model.init(); 

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ============================================================================

	public static Pair<PsoModel, Integer> createCNN_1_L(int workerId, String fileName, int inputDim, int freeze_index) {

		MultiLayerNetwork pretrained = pretrainedModelCIFAR(fileName).asMultiLayerNetwork();

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();

		int start = (int) new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(1 + cfg.FREEZE_INDEX)
			.build().numParams();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(1)
			.build();
		
		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)
				.setFeatureExtractor(freeze_index)			// look at model.summary()
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(inputDim)           // for this TF model: 128
						.nOut(NUM_CLASSES) 
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.biasInit(0.0)
						.build())
				.build();

		model.init(); 

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ============================================================================

	public static Pair<PsoModel, Integer> createCIFAR_CNN_Pretrained_CIFAR_Simpler_v6(int workerId, String fileName, int inputDim) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = pretrainedModelCIFAR(fileName).asMultiLayerNetwork();

		// ============================================================================
		// DL4J needs a FineTuneConfiguration to define updater etc.
		// Use NoOp to prevent optimizer assumptions (since PSO will drive updates).
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.inferenceWorkspaceMode(WorkspaceMode.NONE)
				.build();

		int start = (int) new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(2 + cfg.FREEZE_INDEX)
			.build().numParams();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(2)
			.build();
		
		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)
                .addLayer(new GlobalPoolingLayer.Builder()
						.poolingType(PoolingType.AVG)
						.poolingDimensions(1, 2)
						.build())
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(inputDim)       
						.nOut(NUM_CLASSES)    
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.biasInit(0.0)
						.build())
				.build();

		model.init(); 
	
		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}
    // ===================================================================================================

	public static PsoModel pretrainedModelCIFAR(String fileName) {
		try {
			File f = new File(fileName);

			if (!f.exists()) {
				throw new IllegalStateException("Missing pretrained Keras model: " + f.getAbsolutePath());
			}

			MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(	// these are Keras .h5 files (only in Keras)
					f.getAbsolutePath(),
					false   // enforceTrainingConfig = false (ignore Keras optimizer config)
			);

			return new PsoMultiLayerAdapter(model, true);

		} catch (Exception e) {
			throw new RuntimeException("Failed to import CIFAR-10 Keras .h5 model", e);
		}
	}

    // ===================================================================================================

// 	public static PsoModel pretrainedModelMobileNetV2(String fileName) {
// 		try {
// 			File f = new File(fileName);
// 			if (!f.exists()) {
// 				throw new IllegalStateException("Missing pretrained Keras model: " + f.getAbsolutePath());
// 			}

// 			ComputationGraph base = KerasModelImport.importKerasModelAndWeights(
// 					f.getAbsolutePath(),
// 					false
// 			);
// 			return new PsoGraphAdapter(base);

// 		} catch (Exception e) {
// 			throw new RuntimeException("Failed to import MobileNetV2 base from: " + fileName, e);
// 		}
// 	}

	public static PsoModel pretrainedModelMobileNetV2(String fileName) {
		try {

			cfg.TESTABLE_PRETRAINED_MODEL = false;

			File f = new File(fileName);
			if (!f.exists()) {
				throw new IllegalStateException("Missing pretrained Keras model: " + f.getAbsolutePath());
			}

			ComputationGraph base = KerasModelImport.importKerasModelAndWeights(
					f.getAbsolutePath(),
					false
			);

			FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
					.updater(new NoOp())   // no optimizer
					.inferenceWorkspaceMode(WorkspaceMode.NONE)  
					.build();

			ComputationGraph model = new TransferLearning.GraphBuilder(base)
					.fineTuneConfiguration(ftc)
					.build();

			return new PsoGraphAdapter(model);

		} catch (Exception e) {
			throw new RuntimeException("Failed to import MobileNetV2 base from: " + fileName, e);
		}
	}

	// ===================================================================================================

	public static Pair<PsoModel, Integer> createCifarFromMobileNetV2Base(int workerId, String kerasH5Path) {
		try {
			// 1) Import Keras base (include_top=False)
			ComputationGraph base = KerasModelImport.importKerasModelAndWeights(kerasH5Path, false);

			// start = base params BEFORE adding head
			int start = (int) base.numParams();

			// 2) Freeze ALL layers in the base (feature extractor)
			FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
					.seed(123 + workerId)
					.updater(new NoOp())        // PSO moves weights; no optimizer
					// .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
					// .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
					.build();

			// This is the name of the last output layer of MobileNet. Check model.summary (it also indicates if layer has been frozen or not)
			String featureLayer = "out_relu";	

			ComputationGraph model = new TransferLearning.GraphBuilder(base)
					.fineTuneConfiguration(ftc)
					.setFeatureExtractor(featureLayer) // freeze base up to here

					// GAP: no trainable params
					.addLayer("gap",
							new GlobalPoolingLayer.Builder()
									.poolingType(PoolingType.AVG)
									// pool across spatial dims only; safe for NHWC imports too
									.poolingDimensions(1, 2)
									.build(),
							featureLayer)

					// output head (trainable)
					.addLayer("new_output",
							new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
									// IMPORTANT: for MobileNetV2, channels=1280 after out_relu
									.nIn(1280)
									.nOut(NUM_CLASSES)
									.activation(Activation.SOFTMAX)
									.weightInit(WeightInit.XAVIER)
									.biasInit(0.0)
									.build(),
							"gap")

					.setOutputs("new_output")
					.build();

			model.init();
			// Dl4jParamUtils.printLayerHelpers(model, new long[]{1, 32, 32, 3});

			// safety check: head size must be exactly 1280*numClasses + numClasses
			int expectedHead = 1280 * NUM_CLASSES + NUM_CLASSES;
			int actualHead = (int) model.numParams() - start;
			if (actualHead != expectedHead) {
				throw new IllegalStateException("Head param mismatch. expected=" + expectedHead +
						" actual=" + actualHead + " start(base.numParams)=" + start + " tl.numParams=" + model.numParams());
			}
			// for (org.deeplearning4j.nn.api.Layer l : model.getLayers()) {
			// 	Layer conf = l.conf().getLayer();
			// 	if (conf instanceof ConvolutionLayer) {
			// 		System.out.println(conf.getLayerName() + " algoMode=" +
			// 				((ConvolutionLayer) conf).getCudnnAlgoMode());
			// 	}
			// }

			return Pair.of(new PsoGraphAdapter(model), start);

		} catch (Exception e) {
			throw new RuntimeException("Failed to import and build transfer model from: " + kerasH5Path, e);
		}
	}

	// ===================================================================================================

	public static Pair<PsoModel, Integer> createCifarFromResNet20Stl10(
			int workerId,
			String kerasH5Path
	) {
		try {
			ComputationGraph base = KerasModelImport.importKerasModelAndWeights(kerasH5Path, false);

			int start = (int) base.numParams();

			FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
					.seed(123 + workerId)
					.updater(new NoOp())   
					.build();

			String featureLayer = "global_average_pooling2d"; 

			ComputationGraph model = new TransferLearning.GraphBuilder(base)
					.fineTuneConfiguration(ftc)

					.setFeatureExtractor(featureLayer)
					.removeVertexAndConnections("dense")
					.addLayer("new_output",
							new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
									.nIn(64)                 // <-- GAP output channels
									.nOut(NUM_CLASSES)
									.activation(Activation.SOFTMAX)
									.weightInit(WeightInit.XAVIER)
									.biasInit(0.0)
									.build(),
							featureLayer)

					.setOutputs("new_output")
					.build();

			model.init();

			int oldHead = 64 * 10 + 10; // 650
			int baseNoHead = start - oldHead;

			return Pair.of(new PsoGraphAdapter(model), baseNoHead);

		} catch (Exception e) {
			throw new RuntimeException("Failed to import and build transfer model from: " + kerasH5Path, e);
		}
	}
	// ===================================================================================================

	public static Pair<PsoModel, Integer> createCifarFromMobileNet(int workerId, String kerasH5Path) {
		try {
			// 1) Import Keras base (include_top=False)
			ComputationGraph base = KerasModelImport.importKerasModelAndWeights(kerasH5Path, false);

			// start = base params BEFORE adding head
			int start = (int) base.numParams();

			// 2) Freeze ALL layers in the base (feature extractor)
			FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
					.seed(123 + workerId)
					.updater(new NoOp())        // PSO moves weights; no optimizer
					// .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
					// .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
					.build();

			// This is the name of the last output layer of MobileNet. Check model.summary (it also indicates if layer has been frozen or not)
			String featureLayer = "conv_pw_13_relu";	

			ComputationGraph model = new TransferLearning.GraphBuilder(base)
					.fineTuneConfiguration(ftc)
					.setFeatureExtractor(featureLayer) // freeze base up to here

					// GAP: no trainable params
					.addLayer("gap",
							new GlobalPoolingLayer.Builder()
									.poolingType(PoolingType.AVG)
									// pool across spatial dims only; safe for NHWC imports too
									.poolingDimensions(1, 2)
									.build(),
							featureLayer)

					// output head (trainable)
					.addLayer("new_output",
							new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
									// IMPORTANT: for MobileNetV2, channels=1280 after out_relu
									.nIn(1024)
									.nOut(NUM_CLASSES)
									.activation(Activation.SOFTMAX)
									.weightInit(WeightInit.XAVIER)
									.biasInit(0.0)
									.build(),
							"gap")

					.setOutputs("new_output")
					.build();

			model.init();
			// Dl4jParamUtils.printLayerHelpers(model, new long[]{1, 32, 32, 3});

			return Pair.of(new PsoGraphAdapter(model), start);

		} catch (Exception e) {
			throw new RuntimeException("Failed to import and build transfer model from: " + kerasH5Path, e);
		}
	}

	// ======================================================================================================================

	public static PsoModel pretrainedModelMNIST(String fileName) {
		try {

			File f = new File(fileName);
			
			if (!f.exists()) {
				throw new IllegalStateException("Missing pretrained Keras model: " + f.getAbsolutePath());
			}

			MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(
					f.getAbsolutePath(),
					false   // enforceTrainingConfig = false (not using Keras optimizer config)
			);

			return new PsoMultiLayerAdapter(model, true);

		} catch (Exception e) {
			throw new RuntimeException("Failed to import Keras .h5 model", e);
		}
	}

	// ======================================================================================================================

	public static PsoModel createMNIST_CNN_Pretrained_MNIST_v1(int workerId, String fileName) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = pretrainedModelMNIST(fileName).asMultiLayerNetwork();

		// ============================================================================

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.updater(new NoOp()) 
				.build();

		final int flattenDim = 32 * 5 * 5;  // 800
		MultiLayerNetwork model = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				// .setFeatureExtractor(featureLayer) // freeze base up to here
				.removeLayersFromOutput(2)
				.addLayer(new DenseLayer.Builder()
						.nIn(flattenDim)
						.nOut(64)
						.activation(Activation.RELU) 
						.build())
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(64)
						.nOut(NUM_CLASSES)       
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		return new PsoMultiLayerAdapter(model);
	}
	
	// ===========================================================================================

	public static Pair<PsoModel, Integer>  createMNIST_CNN_Pretrained_MNIST_Simpler_v1(int workerId, String fileName, int inputDim) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = pretrainedModelMNIST(fileName).asMultiLayerNetwork();

		// ============================================================================
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp()) 
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(2)	
			.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc) 
				.addLayer(new DenseLayer.Builder()
					.nIn(inputDim)            
					.nOut(64)
					.activation(Activation.RELU)
					.weightInit(WeightInit.XAVIER)
					.biasInit(0.0)
					.build())
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(64)
						.nOut(NUM_CLASSES)     
						.activation(Activation.SOFTMAX)	// OutputLayer in DL4J contains its own activation function (softmax / sigmoid / etc.)	
														// this depends on the methodology used to define activation layers. They can be embedded or
														// be external (right afterwards) to dense layers
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model, false), start);
	}

	// ===========================================================================================

	public static Pair<PsoModel, Integer>  createCNNModel_1_Layer(int workerId, String fileName, int inputDim) {

		// Pretrained Model ===========================================================

		MultiLayerNetwork pretrained = pretrainedModelMNIST(fileName).asMultiLayerNetwork();

		// ============================================================================
		
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp()) 
				.build();

		int start = (int) new TransferLearning.Builder(pretrained) 
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(1 + cfg.FREEZE_INDEX)
				.build().numParams();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(1)
				.build();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc) 
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(inputDim)
						.nOut(NUM_CLASSES)   
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v4(int workerId, String filename, int inputDim) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();
		
		// ============================================================================
		// DL4J needs a FineTuneConfiguration to define the updater (Adam, SGD, learning rate )
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())   // <-- prevents optimizer assumptions
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(3)	// its 2 because for some reason the activation layers counts as well
			.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)     // <-- REQUIRED in 1.0.0-M2.1
				// .setFeatureExtractor(2)
				.addLayer(new DenseLayer.Builder()
					.nIn(inputDim)           
					.nOut(64)
					.activation(Activation.RELU)
					.weightInit(WeightInit.XAVIER)
					.biasInit(0.0)
					.build())
				.addLayer(new DenseLayer.Builder()
					.nIn(64)         
					.nOut(32)
					.activation(Activation.RELU)
					.weightInit(WeightInit.XAVIER)
					.biasInit(0.0)
					.build())
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(32)
						.nOut(NUM_CLASSES)   
						.activation(Activation.SOFTMAX)	// OutputLayer in DL4J contains its own activation function (softmax / sigmoid / etc.)	
														// this depends on the methodology used to define activation layers. They can be embedded or
														// be external (right afterwards) to dense layers
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();
		return Pair.of(new PsoMultiLayerAdapter(model, true), start);

	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v5(int workerId, String filename, int inputDim) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();
		
		// ============================================================================
		// DL4J needs a FineTuneConfiguration to define the updater (Adam, SGD, learning rate )
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())   // <-- prevents optimizer assumptions
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(3)	// its 2 because for some reason the activation layers counts as well
			.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)     // <-- REQUIRED in 1.0.0-M2.1
				.setFeatureExtractor(5)
				.addLayer(new DenseLayer.Builder()
					.nIn(inputDim)        
					.nOut(64)
					.activation(Activation.RELU)
					.weightInit(WeightInit.XAVIER)
					.biasInit(0.0)
					.build())
				.addLayer(new DenseLayer.Builder()
					.nIn(64)          
					.nOut(32)
					.activation(Activation.RELU)
					.weightInit(WeightInit.XAVIER)
					.biasInit(0.0)
					.build())
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(32)
						.nOut(NUM_CLASSES)    
						.activation(Activation.SOFTMAX)	
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();
		return Pair.of(new PsoMultiLayerAdapter(model, true), start);

	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v6(int workerId, String filename, int inputDim) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();
		
		// ============================================================================
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())  
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(2)
			.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)     // <-- REQUIRED in 1.0.0-M2.1
				.setFeatureExtractor(7)
				.addLayer(new DenseLayer.Builder()
					.nIn(inputDim)            // IMPORTANT
					.nOut(64)
					.activation(Activation.RELU)
					.weightInit(WeightInit.XAVIER)
					.biasInit(0.0)
					.build())
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(64)
						.nOut(NUM_CLASSES)   
						.activation(Activation.SOFTMAX)	
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();
		return Pair.of(new PsoMultiLayerAdapter(model, true), start);

	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v6_1(int workerId, String filename, int inputDim) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();
		
		// ============================================================================
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())  
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(1)
			.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)     // <-- REQUIRED in 1.0.0-M2.1
				.setFeatureExtractor(8)
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(inputDim)
						.nOut(NUM_CLASSES)    
						.activation(Activation.SOFTMAX)	
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();
		return Pair.of(new PsoMultiLayerAdapter(model, true), start);

	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v7(
			int workerId, String filename) {

		MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();

		// Remove conv2d_2 (1x1), GAP, activation  => keep conv stack ending at 7x7x64
		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(3)
				.build();

		int start = (int) truncated.numParams();

		// Add OutputLayer, but MUST flatten CNN activations first via preprocessor
		// After 2x MaxPool: 28->14->7, channels=64  => 7*7*64 = 3136 inputs
		int h = 7, w = 7, c = 64;
		int flattened = h * w * c; // 3136

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(flattened)              // 3136
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.biasInit(0.0)
						.build())
				.setInputPreProcessor(4, new NhwcToFeedForwardPreProcessor(7, 7, 64))
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v7_1(
			int workerId, String filename, int inputDim) {

		MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();

		// Remove conv2d_2 (1x1), GAP, activation  => keep conv stack ending at 7x7x64
		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(3)
				.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
			.fineTuneConfiguration(ftc)
			.addLayer(new GlobalPoolingLayer.Builder(PoolingType.AVG)
				.poolingDimensions(1, 2)   // NHWC: pool H,W
				.collapseDimensions(true)  // default, keeps output [N, C]
				.build())
			.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
				.nIn(inputDim)                 // because conv2d_1 outputs 64 channels
				.nOut(NUM_CLASSES)
				.activation(Activation.SOFTMAX)
				.weightInit(WeightInit.XAVIER)
				.biasInit(0.0)
				.build())
			.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v7_2(
			int workerId, String filename, int inputDim) {

		MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(3)
				.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
			.fineTuneConfiguration(ftc)
			.addLayer(new ConvolutionLayer.Builder(3, 3)	// 28 x 28 x 1
				.nIn(64)
				.nOut(10)       
				.stride(1, 1)
				.padding(0, 0)							// 26 x 26 x 8
				.activation(Activation.RELU)
				.dataFormat(CNN2DFormat.NHWC)  
				.build())
			.addLayer(new GlobalPoolingLayer.Builder(PoolingType.AVG)
				.poolingDimensions(1, 2)   // NHWC: pool H,W
				.collapseDimensions(true)  // default, keeps output [N, C]
				.build())
			.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
				.nIn(10)                 // because conv2d_1 outputs 64 channels
				.nOut(NUM_CLASSES)
				.activation(Activation.SOFTMAX)
				.weightInit(WeightInit.XAVIER)
				.biasInit(0.0)
				.build())
			.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v7_3(
			int workerId, String filename, int inputDim) {

		MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(3)
				.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
			.fineTuneConfiguration(ftc)
			.addLayer(new ConvolutionLayer.Builder(3, 3)	// 28 x 28 x 1
				.nIn(64)
				.nOut(10)       
				.stride(1, 1)
				.padding(0, 0)							// 26 x 26 x 8
				.activation(Activation.RELU)
				.dataFormat(CNN2DFormat.NHWC)  
				.build())
			.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
				.nIn(inputDim)                 // because conv2d_1 outputs 64 channels
				.nOut(NUM_CLASSES)
				.activation(Activation.SOFTMAX)
				.weightInit(WeightInit.XAVIER)
				.biasInit(0.0)
				.build())
			.setInputPreProcessor(5, new NhwcToFeedForwardPreProcessor(5, 5, 10))

			.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v7_4(
			int workerId, String filename) {

		MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();

		// Remove conv2d_2 (1x1), GAP, activation  => keep conv stack ending at 7x7x64
		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(3)
				.build();

		int start = (int) truncated.numParams();

		// Add OutputLayer, but MUST flatten CNN activations first via preprocessor
		// After 2x MaxPool: 28->14->7, channels=64  => 7*7*64 = 3136 inputs
		int h = 7, w = 7, c = 64;
		int flattened = h * w * c; // 3136

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)
				.addLayer(new DenseLayer.Builder()
					.nIn(flattened)
					.nOut(16)                    // try 128 first
					.activation(Activation.RELU)
					.weightInit(WeightInit.XAVIER)
					.build())
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(16)              // 3136
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.biasInit(0.0)
						.build())
				.setInputPreProcessor(4, new NhwcToFeedForwardPreProcessor(7, 7, 64))
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v7_5(
			int workerId, String filename) {

				MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();
		
		int start = (int) new TransferLearning.Builder(pretrained) 
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(2 + cfg.FREEZE_INDEX)
				.build().numParams();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(2)
				.build();

		// Add OutputLayer, but MUST flatten CNN activations first via preprocessor
		// After 2x MaxPool: 28->14->7, channels=64  => 7*7*64 = 3136 inputs
		int h = 7, w = 7, c = 10;
		int flattened = h * w * c; // 3136

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(flattened)              // 3136
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.biasInit(0.0)
						.build())
				.setInputPreProcessor(5, new NhwcToFeedForwardPreProcessor(7, 7, 10))
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}


	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v7_5_1(
			int workerId, String filename) {
				
		MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();
		
		int start = (int) new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(2 + cfg.FREEZE_INDEX)
				.build().numParams();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(2)
				.build();

		// Add OutputLayer, but MUST flatten CNN activations first via preprocessor
		// After 2x MaxPool: 28->14->7, channels=64  => 7*7*64 = 3136 inputs
		int h = 7, w = 7, c = 10;
		int flattened = h * w * c; // 3136

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)
				.addLayer(new DenseLayer.Builder()
					.nIn(flattened)                 // because conv2d_1 outputs 64 channels
					.nOut(16)
					.activation(Activation.SOFTMAX)
					.weightInit(WeightInit.XAVIER)
					.biasInit(0.0)
					.build())
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(16)              // 3136
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
						.biasInit(0.0)
						.build())
				.setInputPreProcessor(5, new NhwcToFeedForwardPreProcessor(7, 7, 10))
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_Pretrained_MNIST_Simpler_v7_6(
			int workerId, String filename, int inputDim) {

		MultiLayerNetwork pretrained = pretrainedModelMNIST(filename).asMultiLayerNetwork();

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();

		// Remove conv2d_2 (1x1), GAP, activation  => keep conv stack ending at 7x7x64
		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
				.fineTuneConfiguration(ftc)
				.removeLayersFromOutput(3)
				.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
			.fineTuneConfiguration(ftc)
			.addLayer(new GlobalPoolingLayer.Builder(PoolingType.AVG)
				.poolingDimensions(1, 2)   // NHWC: pool H,W
				.collapseDimensions(true)  // default, keeps output [N, C]
				.build())
			.addLayer(new DenseLayer.Builder()
				.nIn(inputDim)                 // because conv2d_1 outputs 64 channels
				.nOut(32)
				.activation(Activation.SOFTMAX)
				.weightInit(WeightInit.XAVIER)
				.biasInit(0.0)
				.build())
			.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
				.nIn(32)                 // because conv2d_1 outputs 64 channels
				.nOut(NUM_CLASSES)
				.activation(Activation.SOFTMAX)
				.weightInit(WeightInit.XAVIER)
				.biasInit(0.0)
				.build())
			.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ======================================================================================================================

	public static PsoModel pretrainedModelLeNet() {	// has MNIST weights / was trained on mnist
		// 1) Load pretrained LeNet (MNIST 10-class)
		ZooModel zoo = LeNet.builder()
				.numClasses(10) // MNIST pretrained weights are for 10 classes
				.build();

		MultiLayerNetwork model;
		try {
			model = (MultiLayerNetwork) zoo.initPretrained(PretrainedType.MNIST);
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed to load pretrained LeNet MNIST", e);
		}

		return new PsoMultiLayerAdapter(model);
	}
	
	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_PretrainedLeNet_v1(int workerId) {

		// Pretrained Model ===========================================================
		ZooModel zoo = LeNet.builder().numClasses(10).build();

		MultiLayerNetwork base;
		try {
			base = (MultiLayerNetwork) zoo.initPretrained(PretrainedType.MNIST);
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed to load pretrained LeNet MNIST", e);
		}
		
		// ============================================================================
		// DL4J needs a FineTuneConfiguration to define the updater (Adam, SGD, learning rate )
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())   // <-- prevents optimizer assumptions
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(base)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(2)	// its 2 because for some reason the activation layers counts as well
			.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)     // <-- REQUIRED in 1.0.0-M2.1
				.setFeatureExtractor(7)
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(500)
						.nOut(NUM_CLASSES)     
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model), start);
	}

	// ======================================================================================================================

	public static Pair<PsoModel, Integer> createMNIST_CNN_PretrainedLeNet_v2(int workerId) {

		// Pretrained Model ===========================================================
		ZooModel zoo = LeNet.builder().numClasses(10).build();

		MultiLayerNetwork base;
		try {
			base = (MultiLayerNetwork) zoo.initPretrained(PretrainedType.MNIST);
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed to load pretrained LeNet MNIST", e);
		}
		
		// ============================================================================
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())
				.trainingWorkspaceMode(WorkspaceMode.NONE)
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
				.cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(base)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(5)
			.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)
				.setFeatureExtractor(4)
				.addLayer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
					.name("maxpool2")
					.kernelSize(2, 2)
					.stride(2, 2)
					.build())
				.addLayer(new GlobalPoolingLayer.Builder(PoolingType.AVG).build())
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(50)
						.nOut(NUM_CLASSES) 
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model), start);
	}

// ======================================================================================================================
	
	public static Pair<PsoModel, Integer> createMNIST_CNN_PretrainedLeNet_v3(int workerId) {

		// Pretrained Model ===========================================================
		ZooModel zoo = LeNet.builder().numClasses(10).build();

		MultiLayerNetwork base;
		try {
			base = (MultiLayerNetwork) zoo.initPretrained(PretrainedType.MNIST);
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Failed to load pretrained LeNet MNIST", e);
		}
		
		// ============================================================================
		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())  
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(base)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(5)
			.build();

		int start = (int) truncated.numParams();

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)
				.setFeatureExtractor(4)
				.addLayer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
					.name("maxpool2")
					.kernelSize(2, 2)
					.stride(2, 2)
					.build())
				.addLayer(new GlobalPoolingLayer.Builder(PoolingType.AVG).build())  // -> (N, 50)
				.addLayer(new DenseLayer.Builder()
					.nIn(50)           
					.nOut(64)
					.activation(Activation.RELU)
					.weightInit(WeightInit.XAVIER)
					.biasInit(0.0)
					.build())

				.addLayer(new DenseLayer.Builder()
					.nIn(64)       
					.nOut(32)
					.activation(Activation.RELU)
					.weightInit(WeightInit.XAVIER)
					.biasInit(0.0)
					.build())

				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
					.nIn(32)         
					.nOut(NUM_CLASSES)
					.activation(Activation.SOFTMAX)
					.weightInit(WeightInit.XAVIER)
					.biasInit(0.0)
					.build())
					.build();
		return Pair.of(new PsoMultiLayerAdapter(model), start);
	}

	// ==========================================================================================
	// LeNet Architecture =======================================================================

    // public MultiLayerConfiguration conf() {
    //     MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed)
    //                     .activation(Activation.IDENTITY)
    //                     .weightInit(WeightInit.XAVIER)
    //                     .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
    //                     .updater(updater)
    //                     .cacheMode(cacheMode)
    //                     .trainingWorkspaceMode(workspaceMode)
    //                     .inferenceWorkspaceMode(workspaceMode)
    //                     .cudnnAlgoMode(cudnnAlgoMode)
    //                     .convolutionMode(ConvolutionMode.Same)
    //                     .list()
    //                     // block 1
    //                     .layer(new ConvolutionLayer.Builder()
    //                             .name("cnn1")
    //                             .kernelSize(5, 5)
    //                             .stride(1, 1)
    //                             .nIn(inputShape[0])
    //                             .nOut(20)
    //                             .activation(Activation.RELU)
    //                             .build())
    //                     .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
    //                             .name("maxpool1")
    //                             .kernelSize(2, 2)
    //                             .stride(2, 2)
    //                             .build())
    //                     // block 2
    //                     .layer(new ConvolutionLayer.Builder()
    //                             .name("cnn2")
    //                             .kernelSize(5, 5)
    //                             .stride(1, 1)
    //                             .nOut(50)
    //                             .activation(Activation.RELU).build())
    //                     .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
    //                             .name("maxpool2")
    //                             .kernelSize(2, 2)
    //                             .stride(2, 2)
    //                             .build())
    //                     // fully connected
    //                     .layer(new DenseLayer.Builder()
    //                             .name("ffn1")
    //                             .activation(Activation.RELU)
    //                             .nOut(500)
    //                             .build())
    //                     // output
    //                     .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
    //                             .name("output")
    //                             .nOut(numClasses)
    //                             .activation(Activation.SOFTMAX) // radial basis function required
    //                             .build())
    //                     .setInputType(InputType.convolutionalFlat(inputShape[2], inputShape[1], inputShape[0]))
    //                     .build();

// cnn1: (5*5*1*20)+20 = 520
// cnn2: (5*5*20*50)+50 = 25,050
// ffn1: (7*7*50*500)+500 = 1,225,500
// out : (500*10)+10 = 5,010
// TOTAL = 1,256,080

	// ======================================================================================================================
	// Iris Dataset Model Architecture 

	public static PsoModel createIrisModel(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
	} 

	// Number of weights in the network calculation:  
        // For Hidden Layer 1   => 4 * 16 + 16 (Bias) 
        // For Hidden Layer 2   => 16 * 16 + 16
        // For Output Layer     => 16 * 3 + 3
        // 403 weights all in all

	// ======================================================================================================================

	public static PsoModel createDenseModel_1(int workerId) {	
		if (printModel) System.out.println("Model: createDenseModel_1");  

		Activation act = Activation.SOFTMAX;
		LossFunctions.LossFunction loss; 

		if (NEURAL_OUTPUT == 1) {
			act = Activation.SIGMOID;
			loss = LossFunctions.LossFunction.XENT;   // binary cross-entropy
		} else {
			act = Activation.SOFTMAX;
			loss = LossFunctions.LossFunction.MCXENT; // multi-class cross-entropy
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new OutputLayer.Builder(loss)
						.nIn(NUM_FEATURES)
						.nOut(NEURAL_OUTPUT)
						.activation(act)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model);
	}
	

	// ======================================================================================================================
	// Wine Dataset Model Architecture 

	public static PsoModel createWineModel(int workerId) {
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
        return new PsoMultiLayerAdapter(model);
	} 

	// Number of weights in the network calculation:  
        // For Hidden Layer 1   => 13 * 32 + 32 
        // For Hidden Layer 2   => 32 * 16 + 16
        // For Output Layer     => 16  * 3 + 3
        // 1027 weights 

	// ======================================================================================================================
	// MNIST Dataset Model Architecture 

	public static PsoModel createMNISTModelMLP(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
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

	public static PsoModel createMNISTModelMLPSimple_0(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createMNISTModelMLPSimple_1(int workerId) {
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
						.activation(Activation.TANH) 
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
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createMNISTModelMLPSimple_2(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
	}	// 784 * 10 + 10 = 7850 parameters	 

	// ======================================================================================================================

    public static PsoModel createMNIST5MLP(int workerId) {

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
        return new PsoMultiLayerAdapter(model);
    }
	// Weight Calculation: 
	// 784×128 + 128 = 100,480
	// 128×64 + 64 = 8,256
	// 64×4 + 4 = 260
	// total = 100,480 + 8,256 + 260 = 108,996 weights

	// ======================================================================================================================

	public static PsoModel createMNISTCnn_New_2(int workerId) {

		if (printModel) {
			System.out.println("Using MNIST5 CNN (PSO-feasible)");
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
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createMNISTModelCNNHeavy(int workerId) {

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
				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.padding(0, 0)
						.build())
				.layer(2, new ConvolutionLayer.Builder(3, 3)
						.stride(1, 1)
						.padding(0, 0)
						.nOut(64)
						.activation(Activation.RELU)
						.build())
				.layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.padding(0, 0)
						.build())
				.layer(5, new DenseLayer.Builder()
						.nOut(128)
						.activation(Activation.SIGMOID)
						.build())
				.layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nOut(nOut)
						.activation(Activation.SOFTMAX)
						.build())

				.setInputType(InputType.convolutionalFlat(height, width, channels))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5MLP_Reduced(int workerId) {

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
		return new PsoMultiLayerAdapter(model);
	}

	// 784→64: 784×64 + 64 = 50,176 + 64 = 50,240
	// 64→32: 64×32 + 32 = 2,048 + 32 = 2,080
	// 32→4: 32×4 + 4 = 128 + 4 = 132
	// Total = 50,240 + 2,080 + 132 = 52,452 parameters

	// ======================================================================================================================

    public static PsoModel createMNISTCnn(int workerId) {
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

        return new PsoMultiLayerAdapter(model);
    }

	// ======================================================================================================================
	// MNIST5CNN


	public static PsoModel createMNIST5Cnn(int workerId) {
		if (printModel) {
				System.out.println("Using CNN MNIST5 Model");
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
							.nOut(NUM_CLASSES)                   // MNIST5 => 4 Classes 	16 × 4 + 4 = 68, // this project only uses model.output(...)
							.activation(Activation.SOFTMAX)
							.build())
					.setInputType(InputType.convolutional(28, 28, 1))
					.build();

			MultiLayerNetwork model = new MultiLayerNetwork(conf);
			model.init();
			return new PsoMultiLayerAdapter(model);
	}
	
	// Total = 80 + 1168 + 68 = 1316

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_Simple(int workerId) {
		if (printModel) {
			System.out.println("Using MNIST5 CNN SIMPLE");
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
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New(int workerId) {

		if (printModel) {
			System.out.println("Using MNIST5 CNN");
		}

		int numClasses = NUM_CLASSES;   // MNIST5 => 4, MNIST => 10

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
		return new PsoMultiLayerAdapter(model);
	}

	// Params => k * k = 3 * 3 = 9, conv = nOut*(k*k*nIn + bias)
	// conv1: 32*(9*1+1)=320	// nIn = 1 because 1 channel because grayscale
	// conv2: 64*(9*32+1)=18496
	// conv3: 64*(9*64+1)=36928

	// Flatten means flattening the feature maps => 64 (number of channels) * 3 * 3 (dimensionality of the feature maps) = 576
	// dense: 576 * 64 + 64 = 36928
	// total 320 + 18496 + 36928 + 36928 = 92932 params
	// Reported Dimensionality: 92932

	public static PsoModel createMNIST5Cnn_New_Simpler(int workerId) {

		if (printModel) {
			System.out.println("Using MNIST5 CNN");
		}

		int numClasses = NUM_CLASSES;   // MNIST5 => 4, MNIST => 10

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
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_2(int workerId) {

		if (printModel) {
			System.out.println("Using MNIST5 CNN (PSO-feasible)");
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
		return new PsoMultiLayerAdapter(model, false);
	}

	// Input Layer is always considered: 
	// 9 * 1 (input) * 8 (output) + 8 = 80
	// 9 * 8 (input) * 16 + 16 = 1168
	// 32 * 5 x 5 x 16 + 32= 12832
	// 32 * 4 + 4 = 132
	// 80 + 1168 + 12832 + 132 = 14212 trainable parameters

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_3(int workerId) {

		if (printModel) {
			System.out.println("Using MNIST5 CNN (PSO-feasible)");
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
				.layer(new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.AVG)
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
	
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_4(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)   // 28x28x1 -> 26x26x6
						.nIn(1)
						.nOut(6)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 13x13x6
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)   // -> 11x11x12
						.nOut(12)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 5x5x12
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder()             // 300 -> 16
						.nOut(16)
						.activation(Activation.TANH)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_4_without_2_Dense(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)   // 28x28x1 -> 26x26x6
						.nIn(1)
						.nOut(6)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 13x13x6
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)   // -> 11x11x12
						.nOut(12)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 5x5x12
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_5(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)   // 28x28x1 -> 26x26x6
						.nIn(1)
						.nOut(6)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 13x13x6
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)   // -> 11x11x12
						.nOut(12)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 5x5x12
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new GlobalPoolingLayer.Builder(PoolingType.AVG) // 5x5x12 -> 12
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}
	
	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_6(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(5, 5)   // 28x28x1 -> 24x24x4
						.nIn(1)
						.nOut(4)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 12x12x4
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)   // -> 10x10x8
						.nOut(8)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 5x5x8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder()             // 200 -> 12
						.nOut(12)
						.activation(Activation.TANH)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_7(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(5, 5)   // 28x28x1 -> 14x14x6
						.nIn(1)
						.nOut(6)
						.stride(2, 2)
						.padding(2, 2)
						.activation(Activation.RELU)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)   // 14x14x6 -> 7x7x10
						.nOut(10)
						.stride(2, 2)
						.padding(1, 1)
						.activation(Activation.RELU)
						.build())
				.layer(new GlobalPoolingLayer.Builder(PoolingType.AVG) // 7x7x10 -> 10
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_8(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)   // 28x28x1 -> 26x26x8
						.nIn(1)
						.nOut(8)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 13x13x8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(1, 1)   // 13x13x8 -> 13x13x4
						.nOut(4)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.TANH)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 6x6x4
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new DenseLayer.Builder()             // 144 -> 10
						.nOut(10)
						.activation(Activation.TANH)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_9(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)   // 28x28x1 -> 26x26x8
						.nIn(1)
						.nOut(8)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 13x13x8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(1, 1)   // 13x13x8 -> 13x13x4
						.nOut(4)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.TANH)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 6x6x4
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_10(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(5, 5)   // 28x28x1 -> 24x24x6
						.nIn(1)
						.nOut(8)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 12x12x6
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_11(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(5, 5)   // 28x28x1 -> 24x24x8
						.nIn(1)
						.nOut(8)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 12x12x8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(1, 1)   // 12x12x8 -> 12x12x4
						.nOut(4)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.TANH)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_12(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(5, 5)   // 28x28x1 -> 24x24x8
						.nIn(1)
						.nOut(8)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 12x12x8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)   // 12x12x8 -> 10x10x8
						.nOut(8)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.TANH)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_13(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 12x12x8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createMNIST5Cnn_New_14(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 12x12x8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG) // -> 12x12x8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(28, 28, 1))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}
	// ======================================================================================================================
	// SUSY Dataset Model Architecture 

	public static PsoModel createSUSYModel_SOFTMAX(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
	}

	// Number of weights in the network calculation:  
		// For Hidden Layer 1   => 18 * 128 + 128  
		// For Hidden Layer 2   => 128 * 128 + 128
		// For Output Layer     => 128 * 2 + 2
		// 19202 weights all in all
		// 38018

	
	// ======================================================================================================================
	// SUSY Dataset Model Architecture - Binary Cross Entropy Loss

	public static PsoModel createSUSYModel(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================
	// Bank Dataset Model Architecture 

	public static PsoModel createBankModel(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
	}
	
	// Number of weights in the network calculation:  
        // For Hidden Layer 1   => 53 * 64 + 64  
        // For Hidden Layer 2   => 64 * 64 + 64
        // For Output Layer     => 64 * 1 + 1
        // 7681 weights all in all
		// this is comparable to NN4K

	// ======================================================================================================================

	public static PsoModel createBankModel40K(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
	}
	
	// Number of weights in the network calculation:  
        // For Hidden Layer 1   => 53 * 256 + 256  
        // For Hidden Layer 2   => 256 * 128 + 128
        // For Output Layer     => 128 * 1 + 1
        // 46849 weights all in all
		// this is comparable to NN40K

	// ======================================================================================================================

	public static PsoModel createAdultModel(int workerId) {

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
		return new PsoMultiLayerAdapter(model);
	}

	// Number of weights in the network calculation:
	// For Hidden Layer 1   => 96 * 64 + 64
	// For Hidden Layer 2   => 64 * 64 + 64
	// For Output Layer     => 64 * 1 + 1
	// Total weights = 10433
	// this is comparable to NN40K)

	// ======================================================================================================================
	// COVERTYPE Dataset Model Architecture

	public static PsoModel createCovertypeModel(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
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

	public static PsoModel createHarModel(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
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

	public static PsoModel createDenseModel_4(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
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

	public static PsoModel createPendigitsModelTanh(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createDenseModel_3(int workerId) {

		if (printModel) System.out.println("Using createDenseModel_3");
		Activation act = Activation.SOFTMAX;
		LossFunctions.LossFunction loss; 

		if (NEURAL_OUTPUT == 1) {
			act = Activation.SIGMOID;
			loss = LossFunctions.LossFunction.XENT;   // binary cross-entropy
		} else {
			act = Activation.SOFTMAX;
			loss = LossFunctions.LossFunction.MCXENT; // multi-class cross-entropy
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)     // 16
						.nOut(64)
						.activation(Activation.TANH)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(64)
						.nOut(64)
						.activation(Activation.TANH)
						.build())
				.layer(new OutputLayer.Builder(loss)
						.nIn(64)
						.nOut(NEURAL_OUTPUT)   // 10
						.activation(act)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createDenseModel_2(int workerId) {	// single hidden layer
		if (printModel) System.out.println("Using createDenseModel_2");

		Activation act = Activation.SOFTMAX;
		LossFunctions.LossFunction loss; 

		if (NEURAL_OUTPUT == 1) {
			act = Activation.SIGMOID;
			loss = LossFunctions.LossFunction.XENT;   // binary cross-entropy
		} else {
			act = Activation.SOFTMAX;
			loss = LossFunctions.LossFunction.MCXENT; // multi-class cross-entropy
		}
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)
						.nOut(32)
						.activation(Activation.TANH)
						.build())
				.layer(new OutputLayer.Builder(loss)
						.nIn(32)
						.nOut(NEURAL_OUTPUT)
						.activation(act)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createPendigitsModelSmaller_3(int workerId) {	// no hidden layer just weights connecting input and output layer ...
		if (printModel) System.out.println("Using PenDigits Ultra-Simple Model (no hidden)");

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
						.nIn(NUM_FEATURES)
						.nOut(NEURAL_OUTPUT)
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================
	// WineQuality Dataset Model Architecture

	public static PsoModel createWineQualityModel(int workerId) {

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
		return new PsoMultiLayerAdapter(model);
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

	public static PsoModel createLetterModel(int workerId) {
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
		return new PsoMultiLayerAdapter(model);
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

	public static PsoModel createLetterModel70K(int workerId) {

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
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================
	// CIFAR3

    public static PsoModel createCifar3Model(int workerId) {

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
        return new PsoMultiLayerAdapter(model);
    }

	// ======================================================================================================================

	public static PsoModel createCifar3Model_PSO_Simple(int workerId) {

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
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createCifarCnn_New_13(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 12x12x8
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nOut(NUM_CLASSES)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutional(32, 32, 3))
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model, false);
	}

	// ======================================================================================================================

	public static PsoModel createCifar3Model_New(int workerId) {	// Recommended

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
		return new PsoMultiLayerAdapter(model);
	}
	// 896+18,496+36,928+65,600+195=122,115​
	// Recorded Dimensionality of the output is: 122115. 45% accuracy

	// ======================================================================================================================

	public static PsoModel createCifar3Model_New_Simpler(int workerId) {

		if (printModel) {
			System.out.println("Using CIFAR3 CNN (Keras-style better): 32/64/64 -> Dense(64 relu) -> Softmax(3)");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.inferenceWorkspaceMode(WorkspaceMode.ENABLED)
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
		return new PsoMultiLayerAdapter(model);
	}
	// 896+18,496+36,928+65,600+195=122,115​
	// Recorded Dimensionality of the output is: 122115. 45% accuracy

	// ======================================================================================================================

	public static PsoModel createCifar3Model_New_Simpler_2(int workerId) {

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
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createCifar3Model_New_Simpler_3(int workerId) {

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
		return new PsoMultiLayerAdapter(model);
	}

	// ======================================================================================================================

	public static PsoModel createCifar3Model_New_Simpler_4(int workerId) {

		if (printModel) {
			System.out.println("Using MNIST5 CNN (PSO-feasible)");
		}

		int numClasses = NUM_CLASSES;   // MNIST5 => 4, MNIST => 10

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
		// Dl4jParamUtils.printLayerHelpers(model, new long[]{1, 3, 32, 32});
		return new PsoMultiLayerAdapter(model);
	}
}

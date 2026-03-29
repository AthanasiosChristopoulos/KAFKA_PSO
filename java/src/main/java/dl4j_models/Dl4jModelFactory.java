package dl4j_models;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.CNN2DFormat;
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
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;

import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.LeNet;
import org.deeplearning4j.zoo.PretrainedType;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.common.primitives.Pair;

public class Dl4jModelFactory {
        
	private static final Config cfg = Config.getInstance();
	private static final String DATASET = cfg.DATASET;
    public static final int NUM_FEATURES = cfg.NUM_FEATURES;
    public static final int NUM_CLASSES = cfg.NUM_CLASSES;
    public static final int NEURAL_OUTPUT = cfg.NEURAL_OUTPUT;

	public static final boolean printModel = false;

	// ===========================================================================================

	public static Pair<PsoModel, Integer> createModel(int workerId, boolean preTrained) {
		// System.out.println("DATASET: " + DATASET);
		PsoModel model = null;
		int head_layer_idx = -1;
		Pair<PsoModel, Integer> pair = null;


		// ================================================================	
		if("iris".equals(DATASET)) {
			model = createDenseModel_1(workerId);
			model = createDenseModel_2(workerId);
			model = createDenseModel_3(workerId);

		// ================================================================	

		} else if ("susy".equals(DATASET)) {
			model = createDenseModel_1(workerId);
			model = createDenseModel_2(workerId);
			model = createDenseModel_3(workerId);

		// ========================================================================================

		} else if (DATASET.contains("pendigits")) {

			int version = 1;

			if(cfg.MODEL_VERSION != -1) {
				version = cfg.MODEL_VERSION;
			}
			System.out.println("Pendigits model version: " + version);
			
			if(version == 1) {
				model = createDenseModel_1(workerId);
			} else if(version == 2) {
				model = createDenseModel_2(workerId);
			} else if(version == 3) {
				model = createDenseModel_3(workerId);
			} else if(version == 4) {
				model = createDenseModel_4(workerId);
				model = createDenseModel_4_RELU(workerId);
			}

		// ========================================================================================

		} else if ("winequality".equals(DATASET)) {
			model = createWineQualityModel(workerId);
			model = createDenseModel_1(workerId);
			model = createDenseModel_2(workerId);
			model = createDenseModel_3(workerId);

		// =========================================================================================
		
		} else if ("mnist5".equals(DATASET)) {	// Forward pass cost: CPU => 200ms / GPU => 30ms  

			model = createDenseModel_1(workerId);
			model = createDenseModel_2(workerId);
			model = createDenseModel_3(workerId);
			model = createDenseModel_4(workerId);

			// ===========================================================
			model = createCnn_2_Dense_1(workerId);
			model = createCnn_2_Dense_2(workerId);		// 0.89
			model = createCnn_1_Dense_3(workerId);		// used for experimentation
			model = createCnn_Global_1_Dense_4_v2(workerId);		// 0.915

			// Experimentation with very small models ========================================
			model = createCnn_Global_1_Dense_5(workerId);	
			model = createCnn_1_Dense_6(workerId);		
			model = createCnn_1_Dense_7(workerId); 	
			model = createCnn_1_Dense_8(workerId); 	// this works too well for what it is. that is because of the dimensionality problem.
			model = createDenseModel_1(workerId);

	// ===================================================================================

		} else if ("mnist".equals(DATASET)) {

			cfg.USING_PRETRAINED_MODEL = false;
			model = createCnn_Global_1_Dense_4(workerId);
			model = createCnn_2_Dense_1(workerId);
			model = createCnn_2_Dense_0(workerId);

			model = createCnn_1_Dense_3(workerId); 	// 0.69, after going heavy on it
			model = createCnn_1_Dense_3_v2(workerId); 	// performs worse at 0.66

			// ======================================================================
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
					case 10 -> filename = "../python/pretrained_model/svhn_v2_mnist_best.h5";					
						// 0.756 accuracy after PSO training, 66.4% in the pretrained
					// case 11 -> filename = "../python/pretrained_model/svhn_28x28x1_v4_mnist_final.h5";	
					case 11 -> filename = "../python/pretrained_model/svhn_v4_mnist_final.h5";				
						// 0.82 PSO training, 0.652 pretrained (also 0.8466667)
						// even if pretty slow it will keep improving
						// recommended for svhn version
						// this also needs a lot of workers !!! for accuracy
						// the more complicated the problem, the most important is the accuracy
						
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
						// 77%, recommended for fashion-mnist
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
						24, 25 -> model = loadPretrainedModel(filename);
						default -> throw new IllegalArgumentException("Unknown version: " + version);
					}

				} else {

					switch (version) {
						case -2 -> pair = createMNIST_CNN_PretrainedLeNet_v1(workerId);		// 0.9
						case -1 -> pair = createMNIST_CNN_PretrainedLeNet_v2(workerId);
						case 0 -> pair = createMNIST_CNN_PretrainedLeNet_v3(workerId);

						case 1 -> pair = createCNN_pretrained_1_L(workerId, filename, 64);	// 0.99, fine-tuneable 0.9
						case 2 -> pair = createCNN_pretrained_1_L(workerId, filename, 800);
						case 3 -> pair = createCNN_pretrained_1_L(workerId, filename, 128);		// 0.89
						case 4 -> pair = createCNN_pretrained_3_L_v4(workerId, filename, 50);
						case 6 -> pair = createCNN_pretrained_1_L(workerId, filename, 64);	// 0.71
						case 7 -> pair = createCNN_pretrained_2_L_v7(workerId, filename);	// 0.77
						// case 7 -> pair = createCNN_pretrained_2_L_v7_1(workerId, filename, 64);	// 0.46
						// case 7 -> pair = createCNN_pretrained_2_L_v7_2(workerId, filename, 10); // 0.66	
						// case 7 -> pair = createCNN_pretrained_2_L_v7_3(workerId, filename, 5 * 5 * 10);	
						// case 7 -> pair = createCNN_pretrained_2_L_v7_4(workerId, filename);	// 0.77
						case 7 -> pair = createCNN_pretrained_2_L_v7_5(workerId, filename);	// 0.84, 0.86 with freeze index 1
						// case 7 -> pair = createCNN_pretrained_2_L_v7_5_1(workerId, filename);	// 0.84, 0.86 with freeze index 1
						// case 7 -> pair = createCNN_pretrained_2_L_v7_6(workerId, filename, 64);	// 0.23
						case 8 -> pair = createCNN_pretrained_2_L_v7_5(workerId, filename);	// 0.72% partially frozen, 0.7% fully frozen
						case 9 -> pair = createCNN_pretrained_1_L(workerId, filename, 800); // 80% Partial Freeze
						case 10 -> pair = createCNN_pretrained_1_L(workerId, filename, 576);	
						case 11 -> pair = createCNN_pretrained_1_L(workerId, filename, 90);
						// case 11 -> pair = createCNN_pretrained_1_L_Logits(workerId, filename, 90);
						case 13 -> pair = createCNN_pretrained_1_L(workerId, filename, 90);
						case 12 -> pair = createCNN_pretrained_1_L(workerId, filename, 198);
						case 14 -> pair = createCNN_pretrained_1_L(workerId, filename, 200);
						case 15 -> pair = createCNN_pretrained_1_L(workerId, filename, 800);
						case 16 -> pair = createCNN_pretrained_1_L(workerId, filename, 128);
						case 17 -> pair = createCNN_pretrained_2_L_v1(workerId, filename, 800);	// 0.8, fine-tuneable 0.7
						case 18 -> pair = createCNN_pretrained_1_L(workerId, filename, 400);
						case 19 -> pair = createCNN_pretrained_1_L(workerId, filename, 784);
							// 78%
						case 20 -> pair = createCNN_pretrained_1_L(workerId, filename, 144);
						case 21 -> pair = createCNN_pretrained_1_L(workerId, filename, 36);
						case 22 -> pair = createCNN_pretrained_1_L(workerId, filename, 27);
						case 23 -> pair = createCNN_pretrained_1_L(workerId, filename, 98);
						case 24 -> pair = createCNN_pretrained_1_L(workerId, filename, 98);
						case 25 -> pair = createCNN_pretrained_1_L(workerId, filename, 98);
							// 81.3%

						default -> throw new IllegalArgumentException("Unknown version: " + version);
					}
				}
			}
		// ==============================================================================================

		} else if (DATASET.contains("cifar")) {

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
						// after training => 81%, for cifar5-half	
						// pretrained model => 0.334
						// recommend
					case 17 -> filename = "../python/pretrained_model/cifar10_base_plus_head_v5_half.h5";	
						// cifar10-half => 60%, cifar5-half => 86%
						// Recommended
					case 18 -> filename = "../python/pretrained_model/cifar10_base_plus_head_v6_half.h5";	
						// cifar5-half => 0.74
					case 19 -> filename = "../python/pretrained_model/cifar10_base_plus_head_v7_half.h5";	
						// cifar5-half => 0.8
					case 20 -> filename = "../python/pretrained_model/cifar100_pretrained_base_v4.h5";	
						// cifar100 to cifar5_half: 63% accuracy
						
					default -> throw new IllegalArgumentException("Unknown CIFAR pretrained version: " + version);
				}

				if (preTrained) {
					System.out.println("Using model version: " + version);

					switch (version) {
						case 1, 3, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20 -> model = loadPretrainedModel(filename);
						case 2, 4, 5, 11 -> model = pretrainedModelMobileNetV2(filename);
						default -> throw new IllegalStateException("Unknown ???" );
					}

				} else {

					switch (version) {
						case 1, 3, 10, 14, 15, 20 -> pair = createCNN_pretrained_1_L(workerId, filename, 128);
						case 7, 13, 17 -> pair = createCNN_pretrained_1_L(workerId, filename, 64); 		// 73% cifar10, 91% cifar5
							// cifar 10 trained 75%, cifar 5 optimized 90%
						case 16 -> pair = createCIFAR_CNN_Pretrained_CIFAR_Simpler_v6(workerId, filename, 128);
						case 6 -> pair = createCIFAR_CNN_Pretrained_CIFAR_Simpler_v6(workerId, filename, 200);
						case 9 -> pair = createCNN_pretrained_1_L(workerId, filename, 64); 		// 60% cifar5 (pretrained 0.014)
						case 8, 18 -> pair = createCNN_pretrained_1_L(workerId, filename, 384); 
							// 77% accuracy pretrained, 75% new head
						case 12 -> pair = createCNN_pretrained_1_L(workerId, filename, 64); 

						case 2, 4 -> pair = createCifarFromMobileNetV2Base(workerId, filename); 
						case 11-> pair = createCifarFromResNet20Stl10(workerId, filename);
						case 5 -> pair = createCifarFromMobileNet(workerId, filename);
						case 19 -> pair = createCNN_pretrained_1_L(workerId, filename, 192);

						default -> throw new IllegalStateException("Unknown ???");
					}
				}
			}

		// ==========================================================================

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
	// public static Pair<PsoModel, Integer> createCNN_pretrained_1_L(int workerId, String fileName, int inputDim, int freeze_index) {

	public static Pair<PsoModel, Integer> createCNN_pretrained_1_L(int workerId, String fileName, int inputDim) {

		MultiLayerNetwork pretrained = loadPretrainedModel(fileName).asMultiLayerNetwork();

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
				// .setFeatureExtractor(freeze_index)			// look at model.summary()
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
		MultiLayerNetwork pretrained = loadPretrainedModel(fileName).asMultiLayerNetwork();

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

	// =================================================

	public static PsoModel loadPretrainedModel(String fileName) {
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

	// ===========================================================
	public static PsoModel createMNIST_CNN_Pretrained_MNIST_v1(int workerId, String fileName) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = loadPretrainedModel(fileName).asMultiLayerNetwork();

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

	public static Pair<PsoModel, Integer>  createCNN_pretrained_2_L_v1(int workerId, String fileName, int inputDim) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = loadPretrainedModel(fileName).asMultiLayerNetwork();

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
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model, false), start);
	}

	// ===========================================================================================

	public static Pair<PsoModel, Integer> createCNNModel_1_Layer(int workerId, String fileName, int inputDim) {

		Activation act = cfg.NEED_PROBS ? Activation.SOFTMAX : Activation.IDENTITY;

		// Pretrained Model ===========================================================

		MultiLayerNetwork pretrained = loadPretrainedModel(fileName).asMultiLayerNetwork();

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
				.addLayer(new DenseLayer.Builder()
						.nIn(inputDim)
						.nOut(NUM_CLASSES)   
						.activation(act)
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ===========================================================================================

	public static Pair<PsoModel, Integer>  createCNNModel_1_Layer_Logits(int workerId, String fileName, int inputDim) {

		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = loadPretrainedModel(fileName).asMultiLayerNetwork();

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
				.addLayer(new DenseLayer.Builder()
						.nIn(inputDim)
						.nOut(NUM_CLASSES)
						.activation(Activation.IDENTITY)   // raw logits
						.weightInit(WeightInit.XAVIER)
						.biasInit(0.0)
						.build())
				.build();

		return Pair.of(new PsoMultiLayerAdapter(model, true), start);
	}

	// ==================================================================
	
	public static Pair<PsoModel, Integer> createCNN_pretrained_3_L_v4(int workerId, String filename, int inputDim) {
		// Pretrained Model ===========================================================
		MultiLayerNetwork pretrained = loadPretrainedModel(filename).asMultiLayerNetwork();
		
		// ============================================================================

		FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
				.seed(123 + workerId)
				.updater(new NoOp())  
				.build();

		MultiLayerNetwork truncated = new TransferLearning.Builder(pretrained)
			.fineTuneConfiguration(ftc)
			.removeLayersFromOutput(3)
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
						.activation(Activation.SOFTMAX)
						.weightInit(WeightInit.XAVIER)
    					.biasInit(0.0)
						.build())
				.build();
		return Pair.of(new PsoMultiLayerAdapter(model, true), start);

	}


	// ================================================================================

	public static Pair<PsoModel, Integer> createCNN_pretrained_2_L_v7(
			int workerId, String filename) {

		MultiLayerNetwork pretrained = loadPretrainedModel(filename).asMultiLayerNetwork();

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

	public static Pair<PsoModel, Integer> createCNN_pretrained_2_L_v7_1(
			int workerId, String filename, int inputDim) {

		MultiLayerNetwork pretrained = loadPretrainedModel(filename).asMultiLayerNetwork();

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

	public static Pair<PsoModel, Integer> createCNN_pretrained_2_L_v7_2(
			int workerId, String filename, int inputDim) {

		MultiLayerNetwork pretrained = loadPretrainedModel(filename).asMultiLayerNetwork();

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

	public static Pair<PsoModel, Integer> createCNN_pretrained_2_L_v7_3(
			int workerId, String filename, int inputDim) {

		MultiLayerNetwork pretrained = loadPretrainedModel(filename).asMultiLayerNetwork();

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

	public static Pair<PsoModel, Integer> createCNN_pretrained_2_L_v7_4(
			int workerId, String filename) {

		MultiLayerNetwork pretrained = loadPretrainedModel(filename).asMultiLayerNetwork();

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

	public static Pair<PsoModel, Integer> createCNN_pretrained_2_L_v7_5(
			int workerId, String filename) {

				MultiLayerNetwork pretrained = loadPretrainedModel(filename).asMultiLayerNetwork();

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

		int h = 7, w = 7, c = 10;
		int flattened = h * w * c; 

		MultiLayerNetwork model = new TransferLearning.Builder(truncated)
				.fineTuneConfiguration(ftc)
				.addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)
						.nIn(flattened)          
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

	public static Pair<PsoModel, Integer> createCNN_pretrained_2_L_v7_5_1(
			int workerId, String filename) {
				
		MultiLayerNetwork pretrained = loadPretrainedModel(filename).asMultiLayerNetwork();

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

	public static Pair<PsoModel, Integer> createCNN_pretrained_2_L_v7_6(
			int workerId, String filename, int inputDim) {

		MultiLayerNetwork pretrained = loadPretrainedModel(filename).asMultiLayerNetwork();

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

	// ========================================================================

	public static PsoModel createDenseModel_1(int workerId) {	// no hidden layers
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

	// =====================================================================================

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

		// ==================================================================================

	public static PsoModel createDenseModel_3(int workerId) {	// two hidden layers

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
						.nIn(NUM_FEATURES)     
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
						.nOut(NEURAL_OUTPUT)   
						.activation(act)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model);
	}

	// ============================================================================

	public static PsoModel createDenseModel_4(int workerId) { // 2 hidden layers and more parameters
		if(printModel) {
			System.out.println("Using PenDigits Model");
		}

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)  
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
						.nOut(NEURAL_OUTPUT) 
						.activation(Activation.SOFTMAX)
						.build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		return new PsoMultiLayerAdapter(model);
	}
	// ===================================================================
	// PENDIGITS Dataset Model Architecture

	public static PsoModel createDenseModel_4_RELU(int workerId) {
		if(printModel) {
			System.out.println("Using PenDigits Model");
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
	// For Hidden Layer 1   => 16 * 128 + 128
	// For Hidden Layer 2   => 128 * 128 + 128
	// For Output Layer     => 128 * 10 + 10
	// 19978 weights 

	// =======================================================================

	public static PsoModel createCnn_2_Dense_0(int workerId) {

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

	// =================================================================================

    public static PsoModel createCnn_Global_1_Dense_4(int workerId) {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123 + workerId)
                .weightInit(WeightInit.RELU)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(16)	
                        .stride(1, 1)
                        .padding(1, 1)     
                        .hasBias(false)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new ActivationLayer.Builder() 	
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)		
                        .nOut(32)
                        .stride(1, 1)
                        .padding(1, 1)
                        .hasBias(false)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new ActivationLayer.Builder()	
                        .activation(Activation.RELU)
                        .build())
                .layer(new GlobalPoolingLayer.Builder()		
                        .poolingType(PoolingType.AVG)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.SPARSE_MCXENT)	
                        .nOut(NUM_CLASSES)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(28, 28, 1)) 
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

	// ===================================================================================

	public static PsoModel createCnn_2_Dense_1(int workerId) {

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
						.nOut(32)               
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

	// Input Layer is always considered: 
	// 9 * 1 (input) * 8 (output) + 8 = 80
	// 9 * 8 (input) * 16 + 16 = 1168
	// 32 * 5 x 5 x 16 + 32= 12832
	// 32 * 4 + 4 = 132
	// 80 + 1168 + 12832 + 132 = 14212 trainable parameters


	// ===================================================================

	public static PsoModel createCnn_2_Dense_2(int workerId) {

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

	// ===================================================================
	public static PsoModel createCnn_1_Dense_3(int workerId) {

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

	// ============================================================================

	public static PsoModel createCnn_1_Dense_3_v2(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(3, 3)   // 28x28x1 -> 26x26x6
						.nIn(1)
						.nOut(8)
						.stride(1, 1)
						.padding(0, 0)
						.activation(Activation.RELU)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX) // -> 13x13x6
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(new ConvolutionLayer.Builder(3, 3)   // -> 11x11x12
						.nOut(16)
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

	// ==============================================================================

	public static PsoModel createCnn_Global_1_Dense_4_v2(int workerId) {

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

	// ===========================================================================

	public static PsoModel createCnn_Global_1_Dense_5(int workerId) {

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

	// ==============================================================================

	public static PsoModel createCnn_1_Dense_6(int workerId) {

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

	// ================================================================================

	public static PsoModel createCnn_1_Dense_7(int workerId) {

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


	// ==================================================================

	public static PsoModel createCnn_1_Dense_8(int workerId) {

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

	// ===============================================================================

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

	// ======================================================================
	// WineQuality Dataset Model Architecture

	public static PsoModel createWineQualityModel(int workerId) {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(123 + workerId)
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(NUM_FEATURES)
						.nOut(12)
						.activation(Activation.RELU)
						.build())
				.layer(new DenseLayer.Builder()
						.nIn(12)
						.nOut(9)
						.activation(Activation.RELU)
						.build())
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

}

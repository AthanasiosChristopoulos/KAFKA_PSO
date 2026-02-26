## Data Streaming / Data Pipeline: =============================================================

- “We just keep feeding data and doing backprop forever.” => This almost never happens in practice
    => catastrophic forgetting of data already processed
    => Deep networks with SGD do not behave well in infinite online mode unless heavily constrained.

- Real streaming systems:
    - Pattern A — Windowed retraining (MOST COMMON)
        - Data stream is split into windows (time-based or count-based)
            For each window:
                - train 
                - converge (Loss stops improving / Weights stabilize)
        - old data fades out
    - Pattern B - Periodic Reset


## Regularization: ========================================================================
 
 - techniques that prevent your neural network from memorizing the training data => we need to prevet overfitting
 - overfitting can be viewed from the validation loss - validation loss and accuracy. This isnt training accuracy but it is accuracy coming from test samples.
    => This runs in parallel during the training
 - it forces the network to learn general patterns
 - it sets penalties for large weights: Loss = data loss + λ * penalty(weights)
    => Weight Decay - Penalize Large weights
    => Without regularization, the train accurcay will keep improving, but the test accuracy will significantly worsen (even if it was improving)
    => reqularization penalty isnt included in a forward pass: probs = model.output(X, false); (this is inference mode)
        => would be added only if using DL4J's internal training loop.
 - Another way: Introducing randomness in the training process (Dropout)
    - Stochastic => During training, randomly disables neurons

## Layers: ===============================================================================================

    - Dropout:
        - Dropout is a regularization trick (Helps us achieve regularization):
            During training, it randomly “turns off” a fraction of activations (50%)
            this prevents the network from relying too much on any single neuron and helps reduce overfitting
            Dropout(0.5) roughly means “keep 50% of units” during training (the other 50% are set to 0).
            At inference time, dropout is disabled.

    - Pooling Layers:
        - GlobalAveragePooling2D => Reduce dimensionality to (H,W,C) → (C).
            - this replaces Flatten(). Flatten is better for percision, but will increase massively the number of parameters in the classification layer
        - MaxPooling / AveragePooling => Reduce (H, W) dimensionality in half
        These layers have no parameters, they always do the same thing (fixed reduction)
        
## Funnels / Valeys: ==========================================================================

The landscape of the neural networks is:
    - partially funnel-like locally
    - multi-funnel / multimodal => this means as a function, NNs, dont behave always the same way.

Dense Neural Networks have a loss landscape of sharp directions , many valeys, no correlation betwen weighrs / more parameter independence

CNNs have smoother valeys ? 

## Overfitting ====================================================================

- Overfitting increases with:
    - Model capacity => number of trainable parameters and how easily it can memorize / size of model
    - Dense Layers: This means multiple Dense Layers, especially those with Flatten after a conv layer, connected to conv layers
        - fully-connected layers are very good at memorizing training examples
    - Multiple conv layers can have the same effect

- Data Augmentation is there to help reduce overfitting
    - This is a natural preprcossesing of the data => like flipping / rotation of the image

## Different Types of Neural Networks: ============================================

 - FNN (Feed Forward NN - the basic Neural Network - Basically the Dense Neural Network)
 - CNNs
 - RNNs

## Tensorflow Callbacks: ==========================================================

This is stuff you put as arguments to fit.

 - Early Stopping:
If the validation loss stops improving for 3 epochs in a row (patience), stop training and go back to the best weights we saw (restore_best_weights = True).
This helps to tackle, Training loss always increasing, but validation loss eventually stops improving or gets worse (overfitting)

```python
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(   # Whenever validation loss improves, save the model to disk. (in case PC dies / crashes)
    "mobilenetv2_cifar10.keras",
    monitor="val_loss",
    save_best_only=True
)

```
## CNNs =========================================================================
 
 - Dimensionality after conv layer:
        - output size = floor((N - F + 2 * P) / S) + 1
        - N = input size, F = Filter size (if 3,3 then F = 3), P = padding, S = Stride
        - .padding(0, 0) => P = 0
        - .padding(1,1) => "same", padding(0,0) => "valid"

 - Dimensionality after maxPooling layer:
        - out = floor((N − F + 2 * P) / S)​ + 1
    
 - If input feature Maps / iunput channels is 8 then
    ```java
    .layer(new ConvolutionLayer.Builder(3, 3)
        .nOut(16)
        .stride(1, 1)
        .padding(0, 0))
    ```

    This is 16 Filters of size 3 × 3 × 8 (NOT 3 × 3 × 1):
        => Filters are not per channel
        => Each filter combines all 8 input channels together into one output.
            => Each filter in the second convolution layer must span ALL input channels
            => these are still 16 Filters, but their dimensionality is 3 x 3 x 8 (it must look at 8 previous feature maps at once)
        => Number_of_parameters = (# Dimensionality of Filters) * (# Channels) * (# Next Layer Neurons)
        => filter combines all 8 previous feature maps (feature fusion) together to detect more complex features.
        => Because meaningful patterns in images usually depend on combinations of simpler features, not each one alone.
        => Real patterns are combinations of primitives
 
 - Why are CNNs generally (GD and PSO) harder to train:
    - Vanishing gradients: gradients shrink exponentially as they propagate backward. This slows or even halts learning
    - Deeper models introduce non-convex loss landscapes, volatile landscapes with: plateaus, sharp minima, saddle points
        => sharp means that the landscape isnt "encouraging". It will tell you "you are going the wrong way" (high loss) when you are going the right way
        => gradient descent also relies on loss results - it just treats them more efficiently

 - CNNs may or may not have a final dense / ouputlayer classifier:
    - If they dont, they are called **fully convolutional**. These are non classification models that just extract features (feature extractors). They dont need an input shape of the images, these are just the weights of the filters, not of the classifier.
        - The input shape still needs to be big enough to survive MaxPooling layers (and generall dimensionality reduction)  
        - This is what is meant by include_top = False. The top is the classifier of the CNN. The weights have been already trained with this classifier and now the conv / filter weights will be distributed as pretrained without their classifier (to perform a new task) 
    - If they do (**include_top = True**), the input / the image dimensionality needs to be defined for the network. Otherwise the number of weights / the structure of the dense layer cannot be determined

    - Specify the dense layers after a CNN either by:
        - using both nIn and nOut at every Dense Layer
        - using only nOut (its a dense layer nIn can be infered). Except for the first input, this needs to be specified in this case by:
            - .setInputType(InputType.convolutionalFlat(height, width, channels))

 - conv1x1:
    - layers.Conv2D(num_classes, kernel_size=1, padding="same")
    - For every pixel in the feature map, compute a score for each class
    - behaves very much like a Dense layer applied at every spatial position. (doesnt case about neighbor pixels)
    - Fully convolutional classifier:
        - Conv2D(10, kernel_size=1)
        - GlobalAveragePooling2D()
    - vs Dense Classifier:
        - Flatten()
        - Dense(10)
        
## Transfer Learning ============================================================================
 
 Two Strategies:
 - 1) Feature Extractor (the freeze base is a feature extractor):
    - A model trained on Task A learns general features that are useful for Task B
        - For example a basic image model is trained on ImageNet (cats, dogs, cars, textures)   
    - keep pretrained conv layers fixed and only train the last classification layer
        - [Conv base (frozen)] → [New Dense Head (trainable)]
    - features need to be general (edges, corners, ...)
    - We may need to put many desne layers, because even if in theory, having extracted the high level features classification should be asy (one dense / classificatiobn layer), MobileNet doesnt output perfectly separable features. They are distorted / noisy
        => Be careful, dense layers tend to overfit

 - 2) Fine-tuning:
    - take the pretrained model, keep most layers frozen but unfreeze the last few layers 
        - use small learning rate, this is only fine tuning
     - Not freezing the underlying model:
        - You may do this in case there is a new to adapt to a new dimensionality: 224×224 images => 32×32 images
        - This is problematic when early convolutions downsample too aggressively
        - Not being frozen doesnt mean that they are randomized, they are already at a good starting point just need to adapt a little bit
            => those features arent perfectly separable, need more complex classification layers

 - Improve performance:
    - Choose a different pretrained model
    - Unfreeze / Train more end Layers

 - If original task vs tranfer tasks are similar then transfer will be completed / transfer gap is small
 - Generally better than randomizing weights at the begining of training
 
 - When having different datasets, then transferability is negatively affected by:
     - 1) optimization difficulties related to splitting networks in the middle (where to set the freeze / trained network and the new network)
     - 2) the specialization of higher layer features to the original task (task A) at the expense of performance on the target task (task B).

## ImageNet Datasets: ==================================================================

    - ImageNet is a huge labeled dataset of images organized into thousands of object categories.
    - ~ 14+ million images 
    - Smallest Version: ImageNet-1K = 1,000 classes, ~1.2 million images
    - ImageNet Models are huge and diverse
    
Pretrained ImageNet CNN models, Ranked from simplest to heaviest:
 - Tier 0: LesNet, MobileNetV3Small, MobileNetV2, EfficientNetB0, NASNetMobile

 - Tier 1: MobileNetV3Large, EfficientNetB1, ResNet50 (or ResNet50V2)
    - ResNet18 => It is shallower 18 Layer but has many more parameters, so MobileNetV2 is ~6× cheaper to run
                => Also has the same input dim 224 × 224 × 3 
 - Tier 2: ResNet101, InceptionV3, Xception, DenseNet121

 - Tier 3: heavy 
    - DenseNet169 / DenseNet201, EfficientNetB2 / B3, InceptionResNetV2

 - Tier 4: don’t run on a laptop
    - VGG16 / VGG19 (huge activations + tons of parameters; also slow)
    - EfficientNetB4–B7, NASNetLarge, ResNet152

 - For DL4J:
 LeNet → SimpleCNN → TextGenerationLSTM → FaceNetNN4Small2 → Darknet19 → TinyYOLO → AlexNet → VGG16 → VGG19 → ResNet50 → InceptionResNetV1
 
 - Comparing MobileNetV2 and MobileNetV3Small:
    - use compare_forward_pass_ms.py 
    - They get about the same time
    - Also they have about the same number of layers



## Activation Test Run: =========================================================================
How much memory do activations cost ?

CIFAR Model Activation Memory (Batch Size = 100, float32 = 4 bytes)

------------------------------------------------------------
INPUT
------------------------------------------------------------
Shape: (100, 32, 32, 3)

Elements:
100 * 32 * 32 * 3 = 307,200 => these are 307,200 numbers ... Each of them is a float32, it costs 4 Bytes each number

Memory:
307,200 * 4 = 1,228,800 bytes ≈ 1.17 MB

------------------------------------------------------------
CONV BLOCK 1
------------------------------------------------------------

Conv1 Output
Shape: (100, 32, 32, 32)

Elements:
100 * 32 * 32 * 32 = 3,276,800

Memory:
≈ 12.5 MB


Conv2 Output
Shape: (100, 32, 32, 32)

Elements:
3,276,800

Memory:
≈ 12.5 MB


MaxPool Output
Shape: (100, 16, 16, 32)

Elements:
100 * 16 * 16 * 32 = 819,200

Memory:
≈ 3.1 MB


------------------------------------------------------------
CONV BLOCK 2
------------------------------------------------------------

Conv3 Output
Shape: (100, 16, 16, 64)

Elements:
100 * 16 * 16 * 64 = 1,638,400

Memory:
≈ 6.25 MB


Conv4 Output
Shape: (100, 16, 16, 64)

Elements:
1,638,400

Memory:
≈ 6.25 MB


MaxPool Output
Shape: (100, 8, 8, 64)

Elements:
100 * 8 * 8 * 64 = 409,600

Memory:
≈ 1.56 MB


------------------------------------------------------------
CONV BLOCK 3
------------------------------------------------------------

Conv5 Output
Shape: (100, 8, 8, 128)

Elements:
100 * 8 * 8 * 128 = 819,200

Memory:
≈ 3.13 MB


Conv6 Output
Shape: (100, 8, 8, 128)

Elements:
819,200

Memory:
≈ 3.13 MB


------------------------------------------------------------
GLOBAL AVERAGE POOLING (GAP)
------------------------------------------------------------

Output Shape: (100, 128)

Elements:
100 * 128 = 12,800

Memory:
≈ 0.05 MB


------------------------------------------------------------
TOTAL ACTIVATION MEMORY (FORWARD PASS)
------------------------------------------------------------

Input      :  1.17 MB
Conv1      : 12.50 MB
Conv2      : 12.50 MB
Pool1      :  3.10 MB
Conv3      :  6.25 MB
Conv4      :  6.25 MB
Pool2      :  1.56 MB
Conv5      :  3.13 MB
Conv6      :  3.13 MB
GAP        :  0.05 MB

--------------------------------
TOTAL ≈ 49.6 MB
--------------------------------

This is more close to: 2 * Activations, because of gradients wrt activations, temporary buffers (cuDNN workspaces)
    => Total: 2 * 49.6 MB = 100MB
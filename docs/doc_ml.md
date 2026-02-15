## Data Streaming / Data Pipeline: ===============================================================================================

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

## Funnels / Valeys:    ===============================================================================================

The landscape of the neural networks is:
    - partially funnel-like locally
    - multi-funnel / multimodal => this means as a function, NNs, dont behave always the same way.

Dense Neural Networks have a loss landscape of sharp directions , many valeys, no correlation betwen weighrs / more parameter independence

CNNs have smoother valeys ? 

## Different Types of Neural Networks: ================================================================================
 - FNN (Feed Forward NN - the basic Neural Network - Basically the Dense Neural Network)
 - CNNs
 - RNNs

 ## CNNs =========================================================================
 
  - Dimensionality after conv layer:
        - output size = floor((N - F + 2 * P) / S) + 1
        - N = input size, F = Filter size (if 3,3 then F = 3), P = padding, S = Stride
        - .padding(0, 0) => P = 0
        - padding(1,1) => "same", padding(0,0) => "valid"

 - Dimensionality after maxPooling layer:
        - out = floor((N − F + 2 * P) / S)​ + 1
    
 - If input feature Maps / iunput channels is 8 then
    ```java
    .layer(new ConvolutionLayer.Builder(3, 3)
        .nOut(16)
        .stride(1, 1)
        .padding(0, 0))
    ```
    This is 16 Filters of size 3×3×8 (NOT 3×3×1)
        => Filters are not per channel
        => Each filter combines all 8 input channels together into one output.
        => filter combines all 8 previous feature maps (feature fusion) together to detect more complex features.
        => Because meaningful patterns in images usually depend on combinations of simpler features, not each one alone.
        => Real patterns are combinations of primitives

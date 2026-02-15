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

## DL4J stuff ===============================================================

 - model.init():
    - Allocates and initializes the parameters (weights & biases - using Xavier of whatever WeightInit you set) (everything previously was just defining the model architecture)
    - Creates the internal computation graph structure
    - Allocates memory views for training 

=> TransferLearning.Builder(...).build() automatically calls init() internally for MultiLayerNetwork
=> You can call model.init(); for safety: initialize if not initialized (its always safe)
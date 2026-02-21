
Library  | What it is                                     | Imports
-------- | ---------------------------------------------- |----------------------------
**ND4J** | Numerical computing library (like NumPy)       | org.deeplearning4j
**DL4J** | Deep learning framework (built on top of ND4J) | org.nd4j

- DL4J uses ND4J internally to do all the math.
    - Uses all the INDArray stuff

## DL4J stuff ===============================================================

 - model.init():
    - Allocates and initializes the parameters (weights & biases - using Xavier of whatever WeightInit you set) (everything previously was just defining the model architecture)
    - Creates the internal computation graph structure
    - Allocates memory views for training 

=> TransferLearning.Builder(...).build() automatically calls init() internally for MultiLayerNetwork
=> You can call model.init(); for safety: initialize if not initialized (its always safe)

 - Evaluation won’t give you the full probs array for every sample, because it’s designed to aggregate results, not return raw outputs.  
    - similar to output, but gives an overall ACC / AUC / F1 / ... score (from all batches reduced)
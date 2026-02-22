
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

## DL4J workspaces =================================================================================

 - ND4J workspace as a reusable arena of memory
    - it prioritizes reuse of memory
    - the library allocates once, and then reuses the same buffers over and over.
    - Arena means: they reuse a big chunk of memory by resetting it at the end of a scope/iteration
 - First time accessing the memory in the work space: it grows to whatever size you need.
   => on early iterations it may grow the workspace until it has seen the “worst” case and then keep that memory for reuse.
   => depends on the parameters: AllocationPolicy.OVERALLOCATE, overallocationLimit(2)
 - After that: allocations inside the workspace are basically “bump pointer” allocations (fast).
 - Recycling workspace: When the workspace scope ends, all temporary arrays are considered invalid and the same memory is reused next iteration.
    => at the end of the workspace loop, all INDArrays' memory content is invalidated.
 - Arrays allocated in a workspace are only valid while that workspace is open. When the workspace closes/reset happens, that memory can be reused/overwritten.
 - You can do what you need within a workspace (or spaces), and if you want to get an INDArray out of it (i.e. to move result out of the workspace), you just call INDArray.detach()

Without workspaces:
   allocate → use → free → allocate → use → free → ...

With workspaces:
   allocate once → reuse → reuse → reuse → ...

🔹 WorkspaceMode.SEPARATE
    Training uses separate workspaces for forward and backward pass
    Slightly slower
    Lower peak memory usage

🔹 WorkspaceMode.SINGLE
    Uses one workspace for everything
    Slightly faster
    Higher peak memory usage

🔹 WorkspaceMode.ENABLED => choose the best behavior

    .trainingWorkspaceMode(WorkspaceMode.ENABLED)
    .inferenceWorkspaceMode(WorkspaceMode.ENABLED)  // use this for inference only

 - Every different model instance takes up its own workspace / memory and leaves this allocated (no matter what). 
    - doesnt get unallocated
    - DL4J/ND4J allocates a bunch of GPU memory the first time a model does a forward pass, and that memory stays attached to that model/backend for reuse.
        => This isnt workspace memory (destroyAllWorkspacesForCurrentThread wont affect it)
        => Here, fast conv algorithms allocate large temporary memory buffers once (the first time .output was executed)
        => activation buffers for largest batch
        => Memory allocated per model is persistent after first output

## On output(): ============================================================================================================

 - Runs forward pass inside a workspace
   => wrapping MultiLayerNetwork.output() in your own workspace is not compatible with that internal check. of outputOfLayerDetached()
 - Then DETACHES the result before returning it
 - During a forward pass DL4J creates a workspaces
   => we already do workspaces and the other associated optimizations for you inside of ComputationGraph and MultiLayerNetwork.
 - workspace size grows to max needed size and then stays allocated. It is reused, not freed.
 - When set (like your WorkspaceMode.SINGLE), DL4J will allocate and reuse internal workspaces for activations / intermediates during forward pass.

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

## DL4J Model Building ====================================================================
```java
.addLayer(new GlobalPoolingLayer.Builder(PoolingType.AVG)
    .poolingDimensions(1, 2)   // NHWC: pool H,W, TAKES AS ARGUMENT THE AXES INDEX
    .collapseDimensions(true)  // default, keeps output [N, C]
    .build())
```

GlobalPoolingLayer takes a rank-4 tensor (CNN output) and reduces some axes by averaging to rank 2 tensor.
A 4-rank tensor has: axes = [axes_0, axes_1, axes_2, axes_3 ] 
In Keras: [N, H, W, C] = [batch, height, width, channels]   // .poolingDimensions(1, 2)
In DL4J: [N, C, H, W] = [batch, channels, height, width]    // .poolingDimensions(2, 3)
Global Average Pooling should do: [N, 7, 7, 64]  →  [N, 64]
   => we want to collapse the Height and Width dimensionality not the Channel

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

## On output(): ===============================================================================

 - Runs forward pass inside a workspace
   => wrapping MultiLayerNetwork.output() in your own workspace is not compatible with that internal check. of outputOfLayerDetached()
 - Then DETACHES the result before returning it
 - During a forward pass DL4J creates a workspaces
   => we already do workspaces and the other associated optimizations for you inside of ComputationGraph and MultiLayerNetwork.
   => DL4J opens workspaces internally
   => The MultiLayerNetwork forward pass code literally opens/uses workspaces
 - workspace size grows to max needed size and then stays allocated. It is reused, not freed.
 - When set (like your WorkspaceMode.SINGLE), DL4J will allocate and reuse internal workspaces for activations / intermediates during forward pass.
 - On output(X, false) // false stands fro training false. This should mean activations arent kept for backprop (hopefully) 
vs model.feedForward(X, false); feedForward returns activations per layer (a list/map). That is “explicit activation storage”:

During a forward pass, DL4J still has to compute layer outputs. Those activations exist at least transiently, and on GPU they usually require temporary buffers (cuDNN workspaces + intermediate arrays).   => this means that activations need to be allocated either way
   => idea with two buffers is possible only if your network is a simple chain / an MLP
   => wont work for mobileNet

The question is whether they’re:
   kept (stored for backprop), or
   thrown away immediately after use (inference-style)

✅ output(x,false) = “give me a detached output; therefore no workspace must be open”
✅ output(x,false, ws) = “put output into this workspace; it’s allowed for a workspace to be open”


If there are exactly one workspace per thread, then:
```java
destroyWorkspace(wsRef) // targets one specific workspace instance.
destroyAllWorkspacesForCurrentThread() // nukes every workspace that the workspace manager
```
These two do the same
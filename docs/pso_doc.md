## PSO Theory:

 - Neighborhood, a relation between each particle, must be defined in advance (neighborhood can be implemented by a graph G = {V, E})
    - V = Vertex = Particle, E = Edge = neighborhood relation between particles

 - If a new particle ﬂies beyond the boundary [Xmin, Xmax], the new position will be set as Xmin or Xmax
 - If a new velocity is beyond the boundary [Vmin, Vmax], the new velocity will be set as Vmin or Vmax.


## TensorFlow (Keras) + PySwarms Implementation

TensorFlow gives you easy model definition, provides the “neural forward pass”.
PySwarms gives you a way to train models without gradients, using PSO => provides the outer optimization loop

## This is a PySwarm, non distributed example:

```python

def get_shape(model):   # take the model, records the shape of the weights => like this [(4,4), (4,), (4,3), (3,)]
                        # this is important because internally (during taining) PSO knows only flat vectors (operates on flat numeric arrays)
    weights_layer = model.get_weights() # this adds biases as well
    shapes = []
    for weights in weights_layer:
        shapes.append(weights.shape)
    return shapes   # and return them as is. This is a conversion between model (particle) => weights
    # if our model has a shape like this: [(4,4), (4,), (4,3), (3,)] → shapes is a list with a total of 35 scalars.

def set_shape(weights, shapes): # this takes a flat vector and reshapes it back to the tensors / model weight shape
    new_weights = []
    index = 0
    for shape in shapes:
        if len(shape) > 1:
            n_nodes = np.prod(shape) + index
        else:
            n_nodes = shape[0] + index
        tmp = np.array(weights[index:n_nodes]).reshape(shape)
        new_weights.append(tmp)
        index = n_nodes
    return new_weights

options = {'c1': 0.4, 'c2': 0.8, 'w': 0.4}  # straight from the formula, v = w * v + c1 * r1 * (pbest − x) + c2 * r2 * (gbest − x)
optimizer = GlobalBestPSO(n_particles=25, dimensions=35, options=options, bounds=bounds) # these are bounds for the weights
    # options = come from the formula. dimensios = of the flattend weight vectors (particles)
    # Position: x(t+1) ← x(t) (current position) + v


def evaluate_nn(W, shape, X_train=X_train, Y_train=Y_train):    # W is the swarm (n_particles * dimensions) => 1 weight vector for each particle
                                                                # this is going to evaluate each particle
    results = []
    for weights in W:
        model.set_weights(set_shape(weights, shape)) # sets the weights of the model (usually they change through training over time)
        score = model.evaluate(X_train, Y_train, verbose=0)  # evaluate each particle and store the accuracy
        results.append(1 - score[1])
    return results


cost, pos = optimizer.optimize(evaluate_nn, 15, X_train=X_train, Y_train=Y_train, shape=shape)  # this runs the entire training with epoch = 15
model.set_weights(set_shape(pos,shape)) # pos will be the optimized particle position after the training is completed
score = model.evaluate(X_test, Y_test)

```

-------------------------------------

## Federated Learning:

 - Workers are client based (run / train on client devices, are not server based - decentralized training):
    - Data remains local

 - Local data is non-IID (heterogeneous)
 - Data is processed distributedly 

 - Workers send weights, not data
 - Coordinator averages weights

## Microservice:

 - Independent deployment (just run a worker.jar)
 - Own runtime + own logic (every worker is an independent Kafka Streams instance)
 - Own data (If Stateful, then each microservice stores its own state)
 - Horizontal Scalability

-------------------------------------

### My Simulation does:

0) Load a pretrained model
1) Read from iris-input topic the data (it has Xtrain and Ytrain).
2) perform an evaluation of the model on this data using model.evaluate(Xtrain, Ytrain)
3) Store if pBest (using KTable)
4) Write pBest value on the iris-output topic

----------------------------------

## Alternative 1: Distributed, data parallel PSO Protocol:

1) Initialization of particles, randomize their initial positions + velocities
    => initialize each particle with the same global model architecture (the architecture never changes, only the weights)
    => assign each worker (N WORKERS, working in parallel) 1 particle (this number could vary, worker could be assigned 1...M particles)
    => distribute training data to each particle 

    Repeat this for N_of_training epochs:
        2) Each worker does:
            => Receive current position (weights) for each assigned particle. (Hot-swap model weights)
            => evaluate the current positions using the data:
                <code> fitness = model.evaluate(X_train, Y_train, verbose=0) </code>
            => if this is a personal best fitness, update pBest.
            => sends the pBest position + the fitness value (this is a historic value from all the positions) to the coordinator

        3) 
            => Coordinator receives pBest scores and from them computes the gBest 
            => updates the velocity using the velocity equation of the i-th particle
                <code> v_i_new = w * v_i + c1 * r1 * (pbest_i − x_i) + c2 * r2 * (gbest_i − x_i) </code>
            => updates their position using the velocity 
                <code> x_i_new = x_i + v_i_new </code>
                    => there may be some bounds used for x and v.
            => replaces (v_i, x_i) with (v_i_new, x_i_new) in the coordinators own records, so they dont have to be sent back and forth
                => stores also each particle’s pBest.
            => sends x_i_new to the workers / particles


----------------------------------

## Alternative 2: Distributed, data parallel PSO Protocol:

1) Initialization of particles, randomize their initial positions + velocities
    => initialize each particle with the same global model architecture (the architecture never changes, only the weights)
    => assign each worker (N WORKERS, working in parallel) 1 particle (this number could vary, worker could be assigned 1...M particles)
    => distribute training data to each particle 

    Repeat this until global model converges to an accuracy > 95%:
        2) Each worker does:
            From Coordinator:
                => Receive current global position x_g (weights) and adopts it for each particle.
                => Receive pBests of all particles

            Repeat this for N_BATCH_SIZE:
                => evaluate the current position using the data:
                    <code> fitness = model.evaluate(X_train, Y_train, verbose=0) </code>
                => if this is a personal best fitness, update pBest.
                => update velocity calculate next position x_i_1 using velocity value
                    => <code> v_i_new = w * v_i + (c / M) * sum(i, random * (pBest_i - x_i) </code>

            => sends the current position x_i and the pBest position to the coordinator

        3) The coordinator does:
            => Coordinator receives pBest and updates them
            => Averages x_i of all particles into x_g and sends the x_g to each particle


====================================================================================================================================

## Alternative 3: Distributed, data parallel PSO Protocol:

1) Initialization of particles, randomize their initial positions + velocities
    => initialize each particle with the same global model architecture (the architecture never changes, only the weights)
    => assign each worker (N WORKERS, working in parallel) 1 particle (this number could vary, worker could be assigned 1...M particles)
    => distribute training data to each particle 

    Repeat this until global model converges to an accuracy > 95%:
        2) Each worker does:
                
            Repeat this for N_BATCH_SIZE:
                => evaluate the current position using a batch of data and a loss function (non differentiable) or a fitness function:
                    <code> fitness = model.evaluate(X_train, Y_train, verbose=0) </code>
                => if this is a personal best fitness, update pBest.

                => Receive pBests of all particles (or gBest) from Coordinator
                => update velocity (potentially using new pBests) and calculate next position x_i_1 using new velocity value
                    => <code> v_i_new = w * v_i + (c / M) * sum(j, random * (pBest_j - x_i) </code>
                    => <code> x_i_1 = x_i + v_i_1 </code>
                
            => sends the current position x_i and then return to original loop

        3) The coordinator does:
            => Coordinator receives pBest and updates them (either updates whole pBest list or just gBest), informing the workers
            => Averages x_i of all particles into x_g and use that to evaluate overall performance of the model
                => only if this x_g has a high enough accuracy (higher than desired accuracy) do we conclude training


====================================================================================================================================

## Alternative 3: Distributed, data parallel PSO Protocol:

1) Initialization (by coordinator) of particles, randomize their initial positions + velocities
    => assign each worker (working in parallel) 1...* particles
    => distribute training data to each particle 

Repeat this for N_of_training epochs:
    2) Each worker does:
        Repeat for Y times or use GM to tell when particle has drifted too far:
            => evaluate the current positions using the data:
                <code> fitness = model.evaluate(X_train, Y_train, verbose=0) </code>
                if this is a personal best fitness, update pBest
            => updates their velocity using the velocity equation 
                <code> v = w * v + c1 * r1 * (pbest − x) + c2 * r2 * (gbest − x) </code>
            => updates their position using the velocity 
                <code> X(t+1) = X(t) + v </code>
    Each iteration:

Coordinator sends each worker a subset of particles or just the ones assigned to that worker.

Worker evaluates fitness for its particles (possibly data-parallel inside the worker).

Worker sends back fitness values (and possibly candidate pBest updates).

Coordinator updates pBest, gBest.

Coordinator updates velocities/positions (or broadcasts needed information and lets workers update locally next iteration).

        After this sends the pBest position + the fitness value to the coordinator

    3) 
        - Coordinator receives pBest scores, compares them with each other and sets the gBest 
        - Sends that gBest to all the workers / particles (to update their local copies), to update their equation parameters

====================================================================================================================================

## Alternative 4: Adapted for Geometric Monitoring FIPSO (Fully Informed)

 - Each agent runs updates on x_i and v_i independently / locally (compute new position each time after a time interval t)
    - Let each particle be influenced by all its neighbors best (not just the global one)

 - At synchronization:
    - devices share or contribute their current x_i(t+1) (just before the sync do x_i(t+1) uppdate) and pBest
    - x_i(t+1) are averaged to the global model (often happens in federated enviroments (FedAvg) + we can use some weighted averaging).
        -> both follow the concept of "synchronous averaging"
        -> FedAvg can be developed from a simple average to a weighted average (weights dependend on score)
    - each agent sets x_i(t) = x_g(t), but the The velocity v_i(t+1), its momentum, is maintained
	
public class PredictionBatchProcessor implements Processor<String, String, String, String> {

====================================================================================================================================

## Formulas for PSO / velocity update:

 - Neighbor best (classical PSO):
    - v_i(t + 1) = c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X) + w * v_i(t)

 - Fully informed:
    - v_i(t + 1) = w * v_i(t) + (c / M) * sum_{j=1..M} [ ρ_ij(t) ⊙ (pBest_j - x_i(t)) ]


-------------

```bash

chmod 777 run_streams.sh
dos2unix run_streams.sh
./run_streams.sh --reset

```

consumerProps.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");

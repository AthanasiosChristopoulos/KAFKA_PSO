## PSO Theory:

 - Neighborhood, a relation between each particle, must be defined in advance (neighborhood can be implemented by a graph G = {V, E})
    - V = Vertex = Particle, E = Edge = neighborhood relation between particles

## TensorFlow (Keras) + PySwarms Implementation

TensorFlow gives you easy model definition, provides the “neural forward pass”.
PySwarms (Python Library) gives you a way to train models without gradients, using PSO => provides the outer optimization loop

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


==============================================================================================

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


==============================================================================================

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

==============================================================================================

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

==============================================================================================
## Formulas for PSO / velocity update:

 - Neighbor best (classical PSO):
    - v_i(t + 1) =  w * v_i(t) + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)

 - Fully informed:
    - v_i(t + 1) = w * v_i(t) + (c / M) * sum_{j=1..M} [ ρ_ij(t) ⊙ (pBest_j - x_i(t)) ]


-------------

```bash

chmod 777 run_streams.sh
dos2unix run_streams.sh
./run_streams.sh --reset

```

==============================================================================================
## Local Version of PSO => Neighborhood based

 - Particles get information only from their own neighborhoods best => local_best instead of gBest.
 - Neighbors == Topological Neighbors (doesnt change during a run)
    - v_i(t + 1) =  w * v_i(t) + c1 * r1 * (pbest - X) + c2 * r2 * (lbest - X)
 - Topology:
    - The population is arranged in a ring, for example in 40 particles:
        - 0 — 1 — 2 — 3 — 4 — 5 — ... — 39 — back to 0
    - a neighborhood of six, or three topological neighbors on each side. means that particle i has:
        - i-3, i-2, i-1, i+1, i+2, i+3 as neighbors 
        - Topologically, every particle has its own neighborhood and neighborhoods overlap heavily
 - Number of Neighborhoods == 15% of the number of particles
    - 40 particles => 40 * 0.15 = 6 Neighborhood size + Number of neighborhoods = 40 (once per particle)

 - PSO with a small neighborhood might perform better on complex problems, 
 while PSO with a large neighborhood would perform better on simple problems
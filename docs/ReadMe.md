## =======================================================================
## Run ===================================================================

```bash

# First run docker
docker compose up
docker compose stop

# Normal Training: Start the training with the configuration parameters from .env file
./run_streams.sh

# or manually from:
mvn -q -DskipTests -Dexec.mainClass=pso.Simulation clean compile exec:java

# Experimentation: Runs whatever expriment is selected via the .env file
./run_streams_exp.sh

# or manually from:
mvn -q -DskipTests -Dexec.mainClass=experimentation.Experimentation clean compile exec:java

```

## Run with Docker: ===========================================================

This means that kafka will run on the Docker. a docker-compose.yml is provided for just this 
```bash
docker compose up
docker compose stop
```

## Run with Kafka local Installation: ===========================================================

Having kafka already installed, you should have a server.properties in the installation dir. The log.dir is the most important setting, and determines where records are stored. 
```bash
mkdir -p /mnt/nas_drive/achristopoulos/kafka-kraft
log.dirs=/mnt/nas_drive/achristopoulos/kafka-kraft/logs
```

Then, you can follow this kraft example:
```bash
cd /mnt/nas_drive/achristopoulos/kafka-local    
bin/kafka-storage.sh random-uuid    # outputs a UUID
bin/kafka-storage.sh format \
  --cluster-id <UUID> \
  --config /mnt/nas_drive/achristopoulos/kafka-local/config/kraft/server.properties

```

# htop Alternatives for GPU: ==========================================================

```bash
nvidia-smi	# Confirm if it was successfully installed
watch -n 0.5 nvidia-smi
watch -n 1 -t nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
```

# Where does Docker store records ?
```bash
docker exec -it broker sh -lc       # runs it as a shell inside the docker container   
docker exec -it broker sh -lc 'du -sh /tmp/kafka-logs'
docker exec -it broker sh -lc 'du -sh /tmp/kafka-logs/*'  # show per partition
```

# ===============================================================================
# How to get Keras 2 .h5 files in Ubuntu server enviroment:

We need keras 2 not keras 3 for the .h5 file conversions. This is the standard way on Ubuntu when you need an older Python.

```bash

# 1) Install prerequisites
sudo apt update
sudo apt install -y software-properties-common

# 2) Add deadsnakes
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# 3) Install Python 3.11 + venv
sudo apt install -y python3.11 python3.11-venv python3.11-dev

source ~/venvs/tf215/bin/activate
```

## ============================================================================================
## ============================================================================================
## ============================================================================================
## Generall Aspects / Topics of this Thesis (they are combined with each other):

 - PSO
 - Neural Networks - Models Used for what datasets
 - Kafka / Kafka Streams
 - Filtering / Efficient Communication Protocol
 - Federated Learning

## =====================================================================================================
## Project Architecture Description: ===================================================================

This is my project for PSO, for my thesis. Its purpose is PSO training of Neural Networks used for dataset classification
    => mostly UCI / common datasets of significant number of samples / features 
    => mostly FNN models, but also trying out CNNs as well
    => having the ability to define / use non differentiable functions

The project is build on top of Kafka, Kafka Streams and Python Consumer and Producers. The Kafka service is running on Docker. 
These are the topics that run on Kafka:
    DATA_TOPIC
    TEST_TOPIC
    PBEST_WEIGHTS_TOPIC
    LOCAL_WEIGHTS_TOPIC
    GBEST_WEIGHTS_TOPIC
    PREDICTION_INPUT_TOPIC
    PREDICTION_OUTPUT_TOPIC

The N Kafka Streams workers read from the Data Topic and train on their own local model. The Data Topic holds a partitioned Dataset,
this is how each worker adds to the parallelization of the processing of the training data. Each Worker trains on different partitions,
i.e. different training data from other workers.

Each Kafka Streams Instance is Java. This is why is uses DL4J for inference and generall data / Kafka record processing.

The architecture is build to support two types of PSO:
    - classical PSO / best-of-neighboorhood 
    - fully informed PSO (FIPS)

## Project Implementation Details: ===================================================================
    
    - Java (Maven Project)
    - Kafka / Kafka Streams
    - DL4J 

## Partitioning: ==============================================================================

N_WORKERS < N_PARTITIONS is not a problem, because if N_PARTITIONS = 40, then:
    5 workers ⇒ each gets ~8 partitions (if 40 partitions)
    10 workers ⇒ each gets ~4 partitions
    20 workers ⇒ each gets ~2 partitions

If N_WORKERS > N_PARTITIONS, then #(N_WORKERS - N_PARTITIONS) workers will remain idle / will have 0 partitions assigned.

## Distributed, data parallel PSO Protocol: ===========================================================

- (1) Initialization of particles, randomize their initial positions + velocities
    => initialize each particle with the same global model architecture (the architecture never changes, only the weights)
    => assign each WORKER (N WORKERS, working in parallel) 1 particle (this number could vary, WORKER could be assigned 1...M particles)
    => distribute training data to each particle 

    While True loop (break condition inside this logic):
        - (2) Each WORKER does:
                
            Repeat this for N_BATCHES:
                => evaluate the current position using a batch of data (BATCH_SIZE) and a loss function (non differentiable):
                => if this is a personal best loss, update pBest (personal best weights - model).
                    => communicate also the pBest to PBEST_WEIGHTS_TOPIC, where the coordinator (gBest) or everyone will read it (fully informed)

                => Receive pBests (Fully Informed) or gBest (Neighborhood Best)
                => update velocity and calculate next position x_i_1 using new velocity value
                    => <code> v_i_new = w * v_i + (c / M) * sum(j, random * (pBest_j - x_i) </code>
                    => <code> x_i_1 = x_i + v_i_1 </code>
                
            => sends the current position x_i (for FedAvg) and then return to original loop until data runs out

        - (3) The COORDINATOR does:
            => Coordinator receives pBest and updates gBest, informing the workers
            => Averages x_i of all particles into x_g and use that to evaluate overall performance of the model
                => Only if this x_g has a high enough accuracy (higher than DESIRED_ACCURACY) or if the DATA_TOPIC has been exhausted,
                   does training conclude.
            
    => After training (if we are not in a streaming envirment), the execution of the COORDINATOR doesnt end, since now the global best model will be used for inference of the data in PREDICTION_INPUT_TOPIC.

## =========================================================================
## Finding and Using New Datasets: =========================================

## Input new Dataset - Model:

 - 1) Add new .env variables for DATASET, NUM_FEATURES_{DATASET}, NUM_CLASSES_{DATASET}
 - 2) Update NUM_FEATURES and NUM_CLASSES in Config.java
 - 3) Define new createModel function in Dl4jModelFactory
 - 4) Append in data_producer.py the load_dataset() function an elif
 - 5) Create train and test Kafka Topics and run data_producer.py 

## ===================================================================
## Existing DATASETS: =========================================================

### Iris: ========================================================
    3 Classes
    150 Samples
    4 Features

### Susy: ========================================================

    2 Classes
    Balanced
    80% on Gradient Descent, 72% on PSO
    5000000 Samples

### Winequality: ========================================================

    - NUM_FEATURES_WINEQUALITY = 12, NUM_CLASSES_WINEQUALITY = 2 (red or white wine)
    - 6500 Samples => 6000 Training Samples repeated 37 times (37 epochs) + 500 test samples 
    - 95% accuracy (Fully Informed), 90% accuracy (Neighborhood Best)

### Pendigits:  =========================================================
    
    - Load into Kafka with:
        - from file my-pendigits.tra (my modified union of the train and test dataset) with 10992 samples
        - NUMBER_OF_DATA_REPEATS: 40
        - train size = 10492, test size = 500
            - Number of Kafka Records: 419680
        - Train shape: (10492, 16) classes / y (labels): (0, 9) with counts: [1091 1091 1092 1007 1092 1007 1008 1090 1007 1007]
        
    16 Features, 10 Classes 
    97% on Gradient Descent, 55% on FIPSO, 50% on GBEST
    Evenly Distributed
    Handwriting digit recognition. Features arent the whole picture, but only 8 coordinates in the picture grid in a specific order:
        - x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8
        - If you plotted those points and connected them in order, you’d get a rough sketch of the digit as written.
        - normalized to a 0–100-ish range 
        - Essentially the points in order form the pen trajectory

### MNIST: ====================================================================

    - Grayscale images, very simple image dataset (means (28×28×1).)
        - When flattend there are only 784 features
        
    - Doesnt need a convolutional neural network, because digits are always centered and a pattern will always be at the same location. 
        => MNIST is unusually “linear-friendly” (means it doesnt need CNN, it can be trained by MLP, which is linear): digits are centered, same size, same orientation
        => Background is clean, it is single color
        
    - Forward pass cost: CPU => 200ms / GPU => 30ms  

    - Typical CNN accuracy: >99%
        => MNIST is considered very easy.

### Tranfer Learning comparison based on dataset: =====================

    FMNIST -> MNIST: 75% (if we are being fair)
    - The model learns more complex visual features from Fashion-MNIST, since its the harder dataset
        => edges, shapes, textures, curves, object structure

    - Digits are simpler shapes, so these features still work.
        => Needed to build stronger feature extractor CNN network
    
    MNIST → Fashion-MNIST: 70%
    - model on MNIST learns simpler features
        => strokes, loops, digit curves, simple edges
    - Fashion-MNIST requires more complex CNN features 

### CIRAR10: =======================================================================

- CIFAR-10 labels are: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
- Images are bigger => CIFAR is (32×32×3) more data input / heavy in comparison to  MNIST => has more channels (3x)
- Model complexity vs achievable accuracy on Cifar 10:
| Model               | Accuracy |
| ------------------- | -------- |
| Simple CNN          | 70–80%   |
| Regularized CNN     | 80–90%   |
| ResNet / modern CNN | 92–96%   |


## ===================================================================================
## Theory / PSO Paramaters ===========================================================

However, major disadvantages of BP are its convergence rate is relatively slow and always being trapped at the local minima.

`## Convergence vs Exploration: =========================================================

 - change model
 - change constants => inertia, C1, C2
 - Change VMAX (in this case VMAX_FACTOR) => high VMAX, allows for velocity explosion, particles may fly completely randomly 
        => low VMAX, reduce explration, converge prematurely 
        - Increasing this helps exploration, but going overboard makes convergence impossible 
        - Increasing velocity too much beyond VMAX may also be detrimental
            - velocities will become similar because they will all get clamped in the same way
            - particles will have the same velocity
            - problem happens when number_of_clamps == dimensionality. This may be solvable with CLAMPING_TYPE=NORM
 - increase the number of input data, more batches means more steps / updates  
 - look how velocity amplitude behaves
    - velocity show always start big and then becose smaller
 - Fully Informed seems to be slower, but converging more surely (its always improving)
 - Improve fitness function evaluation => Needs to be less noisy, increase TRAINING_SIZE:
    - If fitness is noisy, pBests / gBest become noisy, and the swarm can wander to a wrong direction.
 - Restricting the social learning aspect to only the gBest makes the original PSO converge fast.

 - Performance (record processing speed - overall data processing time) affects convergence:
    - Increase record processing speed => more updateX in a shorter amount of time + higher data processing speed
    - 1) Faster updateX => Not as reactive to social directive (because of communication latency)
        - Essentially communication latency becomes more substantial / significant
        - This harms convergence, because particle behavior "depends" more on inertia + cognitive accelarators
    - 2) This can still be regulated by adaptive inertia
`
## Premature congvergence: =========================================================

 - trapped in a local optimum if the search environment is complex with numerous local solutions.
 - Evolutionary operators such as selection, crossover, and mutation have been introduced to the PSO to increase the diversity of the population, and to
improve the ability to escape local minima
 - collision-avoiding mechanisms to prevent particles from moving too close to each other
 - Of course, it is hoped in practice that Pm (gBest and pBest) dont remain fixed. A key source of variation is the updating of Pm over time as new points are found in the search space which are better than those previous ones. If this Pm doesnt manage to change as we go towards it => convergence, but it might be premature 

## Increasing N_WORKERS:

- Increasing N_WORKERS adds compute cost and may proove detrimental, for FULLY INFORMED especially
- At the same time, N_WORKERS can help expanding the search space (this is more begenficial for neighborhood best), exploration increases.
- As N_WORKERS increases, number of  data (batches) decreases per worker. This means: number of updates decreases, which means worse less reliable / convergence and number of times reporting current weights (for monitoring) decreases. 
    => As long as INDEPENDENT_DATA_PROCESSING=false => count_updates decreases
- In practice, increasing N_WORKERS is a net positive (both on time and accuracy), as long as:
    - Data per worker doesnt get reduced (happens if data is already plentiful and convergence happens already before data runs out)
        - Stable number of updates
        - Still using INDEPENDENT_DATA_PROCESSING=false
    - Using neighborhoods (so as to not dialute the direction)
    - You can ensure true parallelism between the workers or at least this isnt computationaly too heavy
            
 - Population size / Number of particles:
    - 20 - 50 number of particles
    - trade-off between variety / search space (more particles) and algorithm speed (fewer particles)

## Velocity:  =========================================================

 - “Acceleration Terms” = velocity change terms by addition 
 - Is initialized to have a significant amplitude at the start
 - Inertia parameters should be adjusted so that velocity decreases slowly overtime as swarm converges
 		- velocity like simulated annealing ? Make it reduce over time

 - **INERTIA_FULLY > INERTIA_G_BEST** because we have to make up for extra directional addition in velocity:
    ```java
    float velocity = INERTIA * velocity[k] + C1 * r1 * (pbest[k] - x_i[k]) + C2 * r2 * (gbest[k] - x_i[k]); // 3 accelarations
    float velocity = INERTIA * velocity[k] + socialAggregate[k]       // 2 accelarations

    // some original versions dont have the Inertia term all together. Observed a significant performance decrease when doing so. 
    // FI-PSO is not meant to drop inertia entirely, by itself, FI’s social term is either:
        // too small (means it has prematurely convergenced) or too noisy
    float velocity = C1 * r1 * (pbest[k] - x_i[k]) + C2 * r2 * (gbest[k] - x_i[k]); // 3 accelarations
    ```

 - ## Clamping Velocity: ===================================================

    - Particles' velocities on each dimension are clamped to a maximum velocity Vmax. The velocity on a single dimension is limited to Vmax.
        - if(|V[i]|< Vmax) V[i] = sign(V[i]) * Vmax
    - Vmax parameter (trade-off: local exploitation vs global exploration):
        - Influences how small or large the steps are when moving through the search space (aka search resolution, fineness)
        - Vmax too high: particles might fly past good solutions 
            - global exploration (maybe too much)
        - Vmax too small: Particles will not explore sufficiently beyond locally good regions (trapped in local optima, not enough velocity)
            - local exploitation
    - Set it at about 10-20% of the dynamic range of the variable for every dimensions (each dimension => different dynamic range):
        - dynamic range: xmin - xmax => bounds for variable in a dimension
        - On each dimension => in the original PSO formulation, each dimension can have its own Vmax.
        - Most implementations (and all PSO_NN uses) use the same Vmax for all dimensions.
    - On NNs set xmin=-1, xmax=+1  NN weights do not have a fixed natural range. But [−1,1] is the assumed  an assumed scale, beucase in this amplitude they get initializied)
    - In reality, if you don’t enforce bounds on weights, then choosing xmin/xmax is arbitrary

 - ## C1, C2 Accelaration Constants: ==========================================

    - Comparison between the two: 
        - a relatively high value of c1 causes particles to extremely wander in the search space.
        - arelatively high value of c2 might cause the problem of premature convergence    
    - Strategy (HPSO-TVAC): 
        - At the Beggining of the search: Increase c1, decrease c2 (exploration)
        - At the Ending of the search: Decrease c1, increase c2 (exploitation) 
    - On both: 
        - Low values allow particles to roam far from target regions before being tugged back (by the pBest / gBest)
        - High values result in abrupt movement toward, or past, target regions (pBest / gBest).
        - Set both to 2.0
    - The limits for the two uniform distributions φ1 and φ2 (if c1 * U[0, 1], then c1 = φ1) are usually the same, the total weight is partitioned into two equal components. C1 => exploration, C2 => convergence
    
 - ## Invertia W:  ===================================================================================

    - Three ways of inertia mechanisms: 1) static, 2) change with iteration number (or with time), 3) adaptive inertia 
        - 1) It can be random static as well (randomly choosen but dtatic during the run)
        - 2) linearly-varying inertia weight (LVIW). Here time t == number of iterations, with T == the maximum number of iterations
                => many time variations functions of this 
        - 3) Introduce feedback parameter Ps(t) - the percentage of particles that succeeded to enhance their fitness in the previous iteration:
                => w(t) = (wmax − wmin) * Ps(t) + wmin
                => there is no way to efficiently compute Ps(t) in a distributed enviroment this will not be implemented here 
            => Logic: In PSO, a high inertia w means: keep more of your current velocity / direction (more momentum), since it seems you are getting correct results this direction (be more effected by current trend, rather than social pulls)

    - As originally developed, w often is decreased linearly from about 0.9 to 0.4 during a run.

    - ## Inertia Adaptation during execution: ===================================================

        - Linear logic, from 0.9 to 0.4 across the run
        - Fuzzy Logic / Controller considers:
            - Current gBest fitness: “Are we, hollistically, doing well right now?”
                - If yes, then decrease exploration
            - Current inertia weight: “How exploratory are we currently?”
                - If inertia too low increase exploration
            - Output: Δw (change in inertia weight)
            - Fuzzy: responds gradually to trends, does soft decisions (doesnt change inertia too much)
        
 - # Constriction factor: ===================================================================

    - Problem: There is no mathematical **guarantee convergence**. We need quaranteed convergence
        - Convergence should happen naturally but there is no quarantee.
    - Solution #1: Decrease inertia 
    - Solution #2: Constriction Factor:
        => In a constricted algorithm, the difference between current position and best position tends toward zero over time as velocity gets updated
        => scale the entire velocity update by a single factor K ? Damps all velocity, insuring convergence.
            => vi​(t+1) = K * (vi​(t) + c1​r1​(pbest−xi​) + c2​r2​(gbest−xi​))
        => ϕ = c1 ​+ c2 (total attraction strength), K is dependent on φ via formula only when φ > 4,
            => large φ → strong pull → oscillation / explosion
            => Stable at: c1 = c2 = 2.05, φ = 4.1   => K ≈ 0.729
            =>  to: v = 0.729 v + 1.494 r1 (...) + 1.494 r2 (...)
    - If this is quaranteed then technically no need for Vmax (but Vmax is still helpful in practice)

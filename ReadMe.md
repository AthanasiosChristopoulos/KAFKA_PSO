## =====================================================================================================================
## Run ============================================================================================================

```bash

./run_streams.sh

# or manually from:
mvn -q -DskipTests -Dexec.mainClass=pso.Simulation clean compile exec:java
mvn -q -DskipTests -Dexec.mainClass=evaluate.EvaluateIrisModel clean compile exec:java
mvn -q -DskipTests -Dexec.mainClass=evaluate.ExportDl4jModel clean compile exec:java
```

## Run Docker: ==============================================================================================================

```bash
docker compose up
docker compose stop

docker compose down     # CAREFUL deletes topics ??? 
```
## Formulas for PSO / velocity update:

 - Neighbor best (classical PSO):
    - v_i(t + 1) = c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X) + w * v_i(t)

 - Fully informed:
    - v_i(t + 1) = w * v_i(t) + (c / M) * sum_{j=1..M} [ ρ_ij(t) ⊙ (pBest_j - x_i(t)) ]


```bash

chmod 777 run_streams.sh
dos2unix run_streams.sh
./run_streams.sh --reset

```

## =====================================================================================================================
## Git: ================================================================================================================

```bash

git clone https://github.com/AthanasiosChristopoulos/Kafka_PSO.git
git push https://github.com/AthanasiosChristopoulos/Kafka_PSO.git

git branch
git branch -d branch_name   # Delete a branch
git branch -D branch_name

git log --oneline

git branch -m DL4J-gBest

git clone https://github.com/AthanasiosChristopoulos/WifiDoctor.git 

Creating Repository:
git init
git remote add origin https://github.com/AthanasiosChristopoulos/WifiDoctor.git

Using git:

git add .
git add Documentantion.txt

git commit -m "Your commit message" (--amend)
git commit --amend --no-edit  # Amend (edit last commit, dont create new one) the commit without changing the commit message

Manage commits:
git log (View Commits)
git reset --soft HEAD~1   # Removes commit but keeps changes
git reset --hard HEAD~1   # Removes commit AND changes

git push -u origin main
git push --force origin main (so you dont have to pull first / be up to date)

git checkout main

Codes:
	username: AthanasiosChristopoulos
	password: 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

git rm -r --cached logs
git rm -r --cached target

```
## Related Work: =====================================================================================================
PySwarm:
Standard PSO works this way:
1) Initialize a population of particles with random positions and velocities on d dimensions
2) For each particle, evaluate the desired optimization fitness function in d variables.
3) Compare particle's fitness evaluation with particle's pbest. If current value is better than pbest, then update pbest 
4) Compare fitness evaluation with the population's overall previous best, to update gBest
5) Change the velocity and position of the particle according to the equations (for the id_th partitle):
    Vid = Vid + c * rand() * (pBest_id - x_id) + c2* Rand() * (gBest - x_id)
    xid= xid + Vid 
6) Loop to step (2) until reached a maximum number of iterations (also called generations).

## =====================================================================================================================
## Project Architecture Description: ===================================================================================

This is my project for PSO, for my thesis

The project is build on top of Kafka, Kafka Streams and Python Consumer and Producers. The Kafka service is running on Docker. 
These are the topics that run on Kafka:
    DATA_TOPIC
    TEST_TOPIC
    PBEST_WEIGHTS_TOPIC
    LOCAL_WEIGHTS_TOPIC
    GLOBAL_WEIGHTS_TOPIC
    PREDICTION_INPUT_TOPIC
    PREDICTION_OUTPUT_TOPIC

The N Kafka Streams workers read from the Data Topic and train on their own local model. The Data Topic holds a partitioned Dataset,
this is how each worker adds to the parallelization of the processing of the training data. Each Worker trains on different partitions,
i.e. different training data from other workers.

The architecture is build to support two types of PSO:
    - classical PSO / best-of-neighboorhood 
    - fully informed PSO (FIPS)

## Partitioning: =======================================================================================================

N_WORKERS < N_PARTITIONS is not a problem, because if N_PARTITIONS = 40, then:
    5 workers ⇒ each gets ~8 partitions (if 40 partitions)
    10 workers ⇒ each gets ~4 partitions
    20 workers ⇒ each gets ~2 partitions

If N_WORKERS > N_PARTITIONS, then #(N_WORKERS - N_PARTITIONS) workers will remain idle / will have 0 partitions assigned.

## =====================================================================================================================
## Distributed, data parallel PSO Protocol: 

- (1) Initialization of particles, randomize their initial positions + velocities
    => initialize each particle with the same global model architecture (the architecture never changes, only the weights)
    => assign each WORKER (N WORKERS, working in parallel) 1 particle (this number could vary, WORKER could be assigned 1...M particles)
    => distribute training data to each particle 

    While True loop (break condition inside this logic):
        - (2) Each WORKER does:
                
            Repeat this for N_BATCHES:
                => evaluate the current position using a batch of data (TRAIN_SIZE) and a loss function (non differentiable):
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
          
## PSO Logic: ======================================================================================================================

 - callPredictionsBatch(List<DataMessage> batch)                        // evaluate PSO position
    - Loss Function (some of them Non - Differentiable)
 - updateX(MultiLayerNetwork model, List<float[]> neighborPBestList)    // update PSO position
 - Serialize / Deserialize Topic:
    - modelToFlatList(MultiLayerNetwork model) 
    - updateModel(MultiLayerNetwork model, float[] flat)
 - Exchange pBest / gBest Weight Messages through Kafka Topics

## =================================================================================================================================
## Kafka Message Documentation:

Input pBest-weights-topic:

    Field             | Type
    -----------------------------
    id_worker         | int
    MsgIndex          | String
    accuracy          | float
    loss              | float
    weights           | float[]

Input local-weights-topic:

    Field             | Type
    -----------------------------
    id_worker         | int
    MsgIndex          | String
    accuracy          | float
    loss              | float
    weights           | float[]

Input input-weights-topic:

    Field             | Type
    -----------------------------
    id_worker         | int
    MsgIndex          | String
    accuracy          | float
    loss              | float
    weights           | float[]


## =================================================================================================================================
## Datasets: =======================================================================================================================

## Input new Dataset - Model:

 - 1) Add new .env variables for DATASET, NUM_FEATURES_{DATASET}, NUM_CLASSES_{DATASET}
 - 2) Update NUM_FEATURES and NUM_CLASSES in Config.java
 - 3) Define new createModel function in Dl4jModelFactory
 - 4) Append in data_producer.py the load_dataset() function an elif
 - 5) Create train and test Kafka Topics

 - Extra:
    - update in evaluate_model.py, by adding elifs to functions:
            - NUM_FEATURES and NUM_CLASSES
            - reconstruct_layer_weights_for_keras()
            - build_keras_model()
            - evaluate_model()

## New Dataset - Specifications:

 - Given the specifications i need you to give me a link to where someone showcases his model and the dataset and 
    that he has achieved a 90% accuracy on that Dataset.

 - Needs to be a relatively well known ML dataset.
 - Not too hard, but not as easy as iris, like it should achieve an accuracy of 90% on normal gradient descent with a simple dense NN on tensorflow
    - This is a MUST: Its necessary for accuracy to be 90% and not be evaluated on AUC (Area Under the Curve)
        - It needs to be evaluated with Accuracy

 - Needs to come, not from python, but externally in like a .csv
 - The model that is going to be used on it should have a significant number of weights (100000 - 1000000)
 - Needs to have multiple classes, more than 3 (ideally 4 up to 7), but not too many ... nothing above 10
        - Also, we need even class distribution among the samples
        - softmax / cross entropy loss should ideally be used here 

 - no CNN (like no image recognition)
 - Have enough rows / samples, somewhere around 10000 and above
 - have a reasonable amount of features (not over 100)
 - Also necessary to define those 3 functions, implementing the model on python (tensorflow), with a model that quaranties good accuracy:
        - evaluate_{dataset_name}()
        - load_{dataset_name}_data()
        - build_{dataset_name}_model()
        - run_{dataset_name}() 
        
 - An example of such functions for the dataset "bank_dataset" is detailed here:
 
```python

# ======================================================================
# BANK DATASET
# ======================================================================

def load_bank_data(path="../data/bank-additional-full.csv"):
    print(f"Loading from: {path}")
    df = pd.read_csv(path, sep=';')

    # Label: yes or no to 1 or 0
    y = (df["y"] == "yes").astype(int).values

    X = pd.get_dummies(df.drop(columns=["y"]), drop_first=True).astype(np.float32).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
        # random_state=123 => sudo randomly ordered dataset
        # stratify = y: Split the data so that each class in y appears in the train and test sets 
            # in the same proportion as the original dataset.
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Train shape:", X_train.shape, "Labels:", y_train.shape)
    print("Test  shape:", X_test.shape, "Labels:", y_test.shape)

    return X_train, X_test, y_train, y_test

# ===============================================================================

def build_bank_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model

# ===============================================================================

def run_bank():
    X_train, X_test, y_train, y_test = load_bank_data(path="../data/bank-additional-full.csv")

    model = build_bank_model(input_dim=X_train.shape[1])

    print("\nTraining...")
    history = model.fit(X_train, y_train, validation_split=0.2,epochs=3,batch_size=256,verbose=2)

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    save_model_as_flat_txt(model, path=f"model_serialization/{DATASET}_model_weights.txt")
```


## DATASETS: =========================================================

### Iris:
    3 Classes
    150 Samples
    4 Features

### Susy:
    2 Classes
    Balanced
    80% on Gradient Descent, 72% on PSO
    5000000 Samples

### Bank:
    2 Classes
    Unbalanced 1/10 class_0 vs 9/10 class_1
    90% Gradient Descent, 90% on PSO

    Did the client subscribe to a bank term deposit after the marketing phone calls ?
        - class_0: yes → the client did subscribe (opened a term deposit)
        - class_1: no → the client did not subscribe

### Adult Income:

    2 Classes
    1/3 vs 2/3 Split between classes
    85% on Gradient Descent, 75% on PSO
    If the income is over 50k or not

### Covertype:  ========================================================

    7 Classes
    85% on Gradient Descent, 80% on PSO
    ## Uneven distribution:
        Counts:
            Class_0: 14978
            Class_1: 40106
            Class_2: 1718
            Class_3: 1687
            Class_4: 2045
            Class_5: 1715
            Class_6: 1751

        Percentages:
            Class_0: 23.40%
            Class_1: 62.67%
            Class_2: 2.68%
            Class_3: 2.64%
            Class_4: 3.20%
            Class_5: 2.68%
            Class_6: 2.74%

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


### Pendigits-HALF: ===============================================================================================

    - Load into Kafka with:
        - from file my-pendigits.tra, which after being class filtered, get 10992 samples
        - NUMBER_OF_DATA_REPEATS: 80
        - train size = 5246, test size = 383
            - Number of Kafka Records: 419680
        - Train shape: (5246, 16) classes / y (labels): (0, 4) with counts: [1065 1066 1066  983 1066    0    0]

    16 Features, 5 Classes 
    99% on Gradient Descent, 93% on FIPSO, 85% on GBEST
    Evenly Distributed
    10490 samples on my modified union of the train and test dataset
    Handwriting digit recognition. Features arent the whole picture, but 8 points in a specific order:
        - x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8
        - normalized to a 0–100-ish range 
        - If you plotted those points and connected them in order, you’d get a rough sketch of the digit as written.
        - Essentially the points in order form the pen trajectory

    - // forward pass cost: CPU = 10ms / GPU = 3ms

### HIGGS: ===============================================================================================

The HIGGS dataset comes from high-energy physics
The task is to distinguish Higgs boson events (class 1) from background events (class 0)
    - Binary classification, NUM_CLASSES = 2

Dataset size => 11,000,000 samples, with 28 features

On gradient descent => 75% (~0.75 accuracy / ~0.83 AUC is reasonable on HIGGS, it is noisy and the classes overlap a lot.)

### MNIST: ==========================================================================================================================

    - Grayscale images, very simple image dataset (means (28×28×1).)
        - When flattend there are only 784 features
        
    - Doesnt need a convolutional neural network, because digits are always centered and a pattern will always be at the same location

    - Forward pass cost: CPU => 200ms / GPU => 30ms  

### CIRAR10: ===================================================================================================

    - CIFAR-10 labels are: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
    - Images are bigger => CIFAR is (32×32×3) more data input / heavy in comparison to  MNIST => has more channels (3x)

## CNNs - Image Datasets: ===================================================================================================

    - Even if this seems a small number of parameters / weights, it is much more computationally expensive to apply a forward pass to a CNN, rather than a Dense NN: 
        - weights are reused multiple times in forward pass we are convoluting.
        - In an MLP (Multi-Layer Perceptron), 784 features connect directly to neurons once.
            - MACs == model parameters since we pass them only one time.
        - In a CNN, those 784 pixels are processed repeatedly via sliding kernels.
        
        - Measurements / Experimentation:
            - CNN Forward Pass: 73.49591064453125ms, and overall training time: 58.079 sec
            - DNN Forward Pass: 6.707677841186523ms, and overall training time: 18.501 sec
            - 12x as much
            
        MAC = Multiply–Accumulate (a sum) => in CNNs MACs are much bigger than model parameters:
        - Conv1: MACs ≈ 28 × 28 × 16 × 9 = 112,896 MACs (3 X 3 = 9)
        - Conv2: MACs ≈ 14 × 14 × 32 × 144 = 903,168 MACs (3 X 3 X 16 = 144, since we have more)

## ===================================================================================
## Theory / PSO Paramaters ===========================================================

However, major disadvantages of BP are its convergence rate is relatively slow and always being trapped at the local minima.

## Improve congvergence: =========================================================

 - change model
 - change constants => velocity, inertia, C1, C2
 - increase the number of children
 - look how velocity amplitude behaves
    - velocity show always start big and then becose smaller
 - Fully Informed seems to be slower, but converging more surely (its always improving)
 - Improve fitness function evaluation => Needs to be less noisy, increase TRAINING_SIZE:
    - If fitness is noisy, pBests / gBest become noisy, and the swarm can wander to a wrong direction.

## What to look at for training process: =========================================

 - convergence (the ideal result is located, but the swarm doesnt converge on it)
    - this means the velocity magnitude needs to be decreasing over time => not staying constant / or getting clamped
    - Velocity is initialized with a significant amplitude which should decrease over time since INERTIA < 1
        - Early iterations: exploration-heavy
        - Late phase: stabilization / convergence
    - cognitive Velocity: Distance to of current position to pBest
    - social Velocity: Distance to of current position to gBest (or the other pBests)
 - Is a good result located ? Can it be found ?
 - Are pBest and gBest remaining constant ? is exploration even working ?
 - Trade-off between exploration and convergence
 - The swarm converged on bad solution / local maximum
 - Is low inertia / velocity holding the swarm back from exploring more solutions faster ?
    - is the velocity being clamped / holded back by a limiter ?
 - Increasing N_WORKERS:
    - Increasing N_WORKERS adds compute cost and may proove detrimental, for FULLY INFORMED especially
    - At the same time, N_WORKERS can help expanding the search space (this is more begenficial for neighborhood best)

## Population size / Number of particles: =============================

 - 20 - 50 number of particles
 - trade-off between variety / search space (more particles) and algorithm speed (fewer particles)

## Velocity:  =========================================================

 - “Acceleration Terms” = velocity change terms by addition 
 - Is initialized to have a significant amplitude at the start
 - Inertia parameters should be adjusted so that velocity decreases slowly overtime as swarm converges
 		- velocity like simulated annealing ? Make it reduce over time

 - **W_INERTIA_FULLY > W_INERTIA_G_BEST** because we have to make up for extra directional addition in velocity:
    ```java
    float velocity = W_INERTIA * velocity[k] + C1 * r1 * (pbest[k] - x_i[k]) + C2 * r2 * (gbest[k] - x_i[k]); // 3 accelarations
    float velocity = W_INERTIA * velocity[k] + socialAggregate[k]       // 2 accelarations
    ```

 - ## Clamping Velocity:

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

 - ## C1, C2 Accelaration Constants:

    - Low values allow particles to roam far from target regions before being tugged back (by the pBest / gBest)
    - High values result in abrupt movement toward, or past, target regions (pBest / gBest).
    - Set both to 2.0

 - ## Invertia W:
    - As originally developed, w often is decreased linearly from about 0.9 to 0.4 during a run.

    - ## Inertia Adaptation during execution:
        - Linear logic, from 0.9 to 0.4 across the run
        - Fuzzy Logic / Controller considers:
            - Current gBest fitness: “Are we, hollistically, doing well right now?”
                - If yes, then decrease exploration
            - Current inertia weight: “How exploratory are we currently?”
                - If inertia too low increase exploration
            - Output: Δw (change in inertia weight)
            - Fuzzy: responds gradually to trends, does soft decisions (doesnt change inertia too much)
        
 - # Constriction factor:
    - Problem: There is no mathematical **guarantee convergence**. We need quaranteed convergence
        - Convergence should happen naturally but there is no quarantee.
    - Solution #1: Decrease inertia 
    - Solution #2: Constriction Factor:
        => scale the entire velocity update by a single factor K ? Damps all velocity, insuring convergence.
            => vi​(t+1) = K * (vi​(t) + c1​r1​(pbest−xi​) + c2​r2​(gbest−xi​))
        => ϕ = c1 ​+ c2 (total attraction strength), K is dependent on φ via formula only when φ > 4,
            => large φ → strong pull → oscillation / explosion
            => Stable at: c1 = c2 = 2.05, φ = 4.1   => K ≈ 0.729
            =>  to: v = 0.729 v + 1.494 r1 (...) + 1.494 r2 (...)
    - If this is quaranteed then technically no need for Vmax (but Vmax is still helpful in practice)

 - # Randomness Dimensionality: 
    ```java
    // 1) static randmoness per updateX / Statistically independent dimensions
    float velocity = W_INERTIA * velocity[k] + C1 * r1 * (pbest[k] - x_i[k]) + C2 * r2 * (gbest[k] - x_i[k]);   // dimension randomness is different per dimension 
    
    // 2) for each dimensions a different random Number is generated / Statistically dependent (coupled) dimensions:
    float velocity = W_INERTIA * velocity[k] + C1 * r1[k] * (pbest[k] - x_i[k]) + C2 * r2[k] * (gbest[k] - x_i[k]);  // dimension randomness is the same
    ```
    - (1) larger search space / converges less easily / less direction / Each weight can “wiggle” independently
        - May struggle comparatively ro rotated version of the problems. 
        - A problem is rotated if:
            1) the directions of improvement are not aligned with axes
            2) progress requires coordinated changes across many variables
            => Can you optimize by adjusting x or y (each dimension) independently ?
    - (2) smaller search space / Particle moves along a fixed, but randomly scaled, ray toward pbest / gbest
    - ## Rotation in NNs:
        - NN Datasets arent explicitly rotated, but neural nets locally behave like rotated problems, which is why stabilization matters more than rotation-invariance tricks. 

## ==================================================================================
## Functional Requirements: =========================================================

### Filtering / Communication prevention:

    - This is an asynchronous protocol. We dont have the ability to stop pipeline / incoming data from Kafka topics
    - But we have the ability for workers / coordinator to choose not to send messages, when it is predicted to be unnecessary 

    - Given the Kafka Topology, we can influence only the communication pipelines 2_1, 2_2 and 6:
        - 2_2 is the federated learning pipeline. It is used to extract the average model, but this doesnt actively partiticipate in PSO.
            - This should be selected as the users preference, since through this form of communication the progression of the algorithm is reported (CLI)

        - 2_1 and 6 are directly used for PSO (pBest / gBest weight messages). 
            - We can choose not to send them if the loss of the new model wasnt significantly improved 

## Non Functional Requirements: =========================================================

	- θελουμε καλο accuracy γρηγορα (trade off) δηλαδη τα δεδομενα πρεπει να επεξεργαζονται γρηγορα για να ειναι streaming περιβαλλον

## ==========================================================================
## Experimentation: =========================================================

 - Load 400000 messages / samples to Kafka Input topic (make the reperation number just high enough for this)
    - divide this by 40 partitions => each partition gets 10000 samples

 - Need to measure:
    - Diagramm 1: y-axis: Accuracy - N_Workers
    - Diagramm 2: y-axis: Training Time - N_Workers

 - Run for N_Workers = [5, 10, 15, 20]
 
 - Performance Measurements:
    - htop (shows logical CPUs and ||| represent CPU time usage on each logical CPU)
        - 6 physical cores × 2 threads = 12 logical CPUs
        - 12 physical cores × 1 thread = 12 logical CPUs
        - use <code>lscpu</code> to evaluate CPU number and number of threads
            - number of physical cores: Core(s) per socket: 6
            - number of threads on its core: Thread(s) per core: 2

    - ## GPU: ================================================================================

        The GPU is capable of executing CNN forward pass faster than the CPU. 
        GPU is only used for DL4J stuff (wherever we are handling DL4J):
            - INDArray probs = model.output(X, false);      # Primary use of the GPU, forward passes => convolutions, matrix multiplications
            1) [CPU parsing + copying]
            2) [CPU → GPU transfer] (Copies data from CPU RAM → GPU VRAM, this is a memory transfer)
                => Memory copy overhead
            3) [GPU forward pass]
            4) [GPU → CPU sync]     (Pull data back to CPU, synchronizing on every batch => get results when they are needed)
                => normally .output() would be async but in this case we need the results immidiatly to calculate loss
                => Sync means => CPU is blocked until the results from the GPU have arrived                  
            5) [CPU loss + accuracy loops]  
        (2) + (4) are overhead (+ GPU scheduling / Kernel launch). If the forward pass cost is small either way, then its not worth it to use GPU, it will end up costing more time.
        The forward pass cost is also determined by batch size, but this needs to be kept small for PSO not to run out of data.
            => On a simple NN, cpu is preferable

        But GPU is limited, it will only help in neural net compute.
            - No contribution for serialization / Kafka messages
            - This means me may be able to afford bigger models or batches, but the primary bottleneck will still be Kafka / CPU Scheduling
            - CUDA just does faster tensor math, convolutions, matrix multiplications
        GPU gets more benefit from increased batch size
        
        - nvidia-smi -l 1
        - <code>nvidia-smi -q</code>  // see gpu specs
        
        # htop Alternatives for GPU: ==========================================================
        - nvidia-smi	# Confirm if it was successfully installed
        - watch -n 0.5 nvidia-smi

 - Performance Theory:
    - True parallelism comes from CPU cores
    - Swap = “RAM overflow to disk” (very slow), if Swap and RAM is huge => problem

 - Kill Zombie Java processes:
```bash
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -n 25   # detect them
sudo pkill -2 java
sudo pkill -9 -f java
```

## =======================================================================================================================

Καταχώριση πρακτικής άσκησης στο φοιτητολόγιο
Είχα κάνει την πρακτική μου άσκηση κατά τους μήνες Ιούλιο–Αύγουστο και μέχρι στιγμής δεν εμφανίζεται στο φοιτητολόγιο.
Επικοινώνησα με το Γραφείο Πρακτικής Άσκησης, το οποίο με ενημέρωσε ότι θα πρέπει να απευθυνθώ σε εσάς.
Θα έπρεπε να έχει ανέβει η πρακτική άσκηση στο φοιτητολόγιο ή μήπως έχω καταλάβει κάτι λάθος;
Αριθμός Μητρώου: 2022030077
Με εκτίμηση,
Αθανάσιος Χριστόπουλος
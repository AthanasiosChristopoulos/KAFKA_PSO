## =======================================================================
## Run ===================================================================

```bash

./run_streams.sh

# or manually from:
mvn -q -DskipTests -Dexec.mainClass=pso.Simulation clean compile exec:java
mvn -q -DskipTests -Dexec.mainClass=evaluate.EvaluateIrisModel clean compile exec:java
mvn -q -DskipTests -Dexec.mainClass=evaluate.ExportDl4jModel clean compile exec:java
```

## Run Docker: ===========================================================
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

## ===========================================================================
## Git: ======================================================================

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

## Related Work: =============================================

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

## Non DIfferentiable Loss Functions: =======================================================

 - Review - Ranking:
    - ZERO_ONE is too discrete. Losses out on information.
    - MAE is middle, continious probability p depended, but it doesnt reward confidence as well as CE its not the ideal function
    - MAE
    - ZERO_ONE < MAE < TOP-K < CROSS_ENTROPY (standard)
 - Smoothness / a dense signal is important because you want PSO to detect loss differences when weights change, even a little. 
 - If its a discrete signal, then PSO gets no gradient like guidance, weights change and loss remains the same flat (not informative, doesnt give a direction)


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
    GLOBAL_WEIGHTS_TOPIC
    PREDICTION_INPUT_TOPIC
    PREDICTION_OUTPUT_TOPIC

The N Kafka Streams workers read from the Data Topic and train on their own local model. The Data Topic holds a partitioned Dataset,
this is how each worker adds to the parallelization of the processing of the training data. Each Worker trains on different partitions,
i.e. different training data from other workers.

Each Kafka Streams Instance is Java. This is why is uses DL4J for inference and generall data / Kafka record processing.

The architecture is build to support two types of PSO:
    - classical PSO / best-of-neighboorhood 
    - fully informed PSO (FIPS)

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
          
## PSO Logic: ===============================================================

 - callPredictionsBatch(List<DataMessage> batch)                        // evaluate PSO position
    - Loss Function (some of them Non - Differentiable)
 - updateX(MultiLayerNetwork model, List<float[]> neighborPBestList)    // update PSO position
 - Serialize / Deserialize Topic:
    - modelToFlatList(MultiLayerNetwork model) 
    - updateModel(MultiLayerNetwork model, float[] flat)
 - Exchange pBest / gBest Weight Messages through Kafka Topics

## Kafka Message Documentation: =======================================================


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

## ===================================================
## ===================================================
## ===================================================
## Datasets: =========================================

## Input new Dataset - Model:

 - 1) Add new .env variables for DATASET, NUM_FEATURES_{DATASET}, NUM_CLASSES_{DATASET}
 - 2) Update NUM_FEATURES and NUM_CLASSES in Config.java
 - 3) Define new createModel function in Dl4jModelFactory
 - 4) Append in data_producer.py the load_dataset() function an elif
 - 5) Create train and test Kafka Topics and run data_producer.py 

 - Extra:
    - update in evaluate_model.py, by adding elifs to functions:
            - NUM_FEATURES and NUM_CLASSES
            - reconstruct_layer_weights_for_keras()
            - build_keras_model()
            - evaluate_model()

## New Dataset - Specifications: ==================================================================

 - Given the specifications i need you to give me a link to where someone showcases his model and the dataset and that he has achieved a 90% accuracy on that Dataset.

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

## ===================================================================
## DATASETS: =========================================================
## Small / Easy Datasets / UCI datasets:
Few Samples / Few classes (2 - 3 classes)

Horse => 300
Vertebral => 310
Wine => 178
Diabetes => 768
Blood Tissue => 748
Seed => 210
Heart => 300
Iris => 150

### Iris: ========================================================
    3 Classes
    150 Samples
    4 Features

### Susy: ========================================================
    2 Classes
    Balanced
    80% on Gradient Descent, 72% on PSO
    5000000 Samples

### Bank: ========================================================
    2 Classes
    Unbalanced 1/10 class_0 vs 9/10 class_1
    90% Gradient Descent, 90% on PSO
    45k samples
        
    Did the client subscribe to a bank term deposit after the marketing phone calls ?
        - class_0: yes → the client did subscribe (opened a term deposit)
        - class_1: no → the client did not subscribe

### Adult Income: ========================================================

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


### Pendigits-HALF: ====================================================================

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

### Letter: =============================================================================================

 - Evenly Distributed
 - Very similar to Pendigits, just letters / alphabetical characters instead of numerical Digits
 - 24 classes (alphabet)

### HIGGS: ===============================================================================================

The HIGGS dataset comes from high-energy physics
The task is to distinguish Higgs boson events (class 1) from background events (class 0)
    - Binary classification, NUM_CLASSES = 2

Dataset size => 11,000,000 samples, with 28 features

On gradient descent => 75% (~0.75 accuracy / ~0.83 AUC is reasonable on HIGGS, it is noisy and the classes overlap a lot.)

### MNIST: ====================================================================

    - Grayscale images, very simple image dataset (means (28×28×1).)
        - When flattend there are only 784 features
        
    - Doesnt need a convolutional neural network, because digits are always centered and a pattern will always be at the same location. 
        => MNIST is unusually “linear-friendly” (means it doesnt need CNN, it can be trained by MLP, which is linear): digits are centered, same size, same orientation
        => Background is clean, it is single color
        
    - Forward pass cost: CPU => 200ms / GPU => 30ms  

### FASHION-MNIST: ====================================================================

    - Train - Samples: 60000, Features => 28 x 28 x 1, 784 features flattend
    - 10 classes, different types of clothing

### CIRAR10: ========================================================================

    - CIFAR-10 labels are: 0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog, 6 frog, 7 horse, 8 ship, 9 truck
    - Images are bigger => CIFAR is (32×32×3) more data input / heavy in comparison to  MNIST => has more channels (3x)

## ================================================================================================
## CNNs - Image Datasets: =========================================================================

    - Even if this seems a small number of parameters / weights, it is much more computationally expensive to apply a 
        forward pass to a CNN, rather than a Dense NN: 
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

    - Specify the dense layers after a CNN either by:
        - using both nIn and nOut at every Dense Layer
        - using only nOut (its a dense layer nIn can be infered). Except for the first input, this needs to be specified in this case by:
            - .setInputType(InputType.convolutionalFlat(height, width, channels))

## =====================================================================================
## PSO friendly Neural Networks Architectures ========================================== 

 - Shallow networks / Low Parameter / Weight Count / Small number of neurons => Low Dimensionality, important for PSO 
    => sensitive to parameter count - PSO performance drops super fast as D increases
    => In backpropagation the size of the model is - overfitting discounted - a net positive
    => PSO obtains the best solution from particles’ interaction, but through high-dimensional search space, 
        it converges at a very slow speed towards the global optimum / usually just fails to find the optimum.
    => prefers Single hidden layer networks
 - This is important because:
    - local optima trap, 
    - but also the potential fluctuation of the velocities of particles such that the successive range of trials is bounded within a sub-plain of the whole search hyper-plain

 - Smooth activation Functions (for dense layer neurons):
    - sigmoid (Range: [0, 1])
    - tanh (Range: [-1, 1])
    - ReLU (might be unstable)

 - No randomness / state allowed: 
    - No BatchNorm layers and Dropout Layers (adds state, but each position should be stateless)
    - fitness(w)=random in that case, at least it wont be deterministic
    - PSO needs to compare different solutions. These solutions need to be as reliable as possible
    - Imagine a gBest / pBest happend because of randomness => randomness makes comparisons unreliable
 - From tests, it has been determined in multiple cases that lowering the size of the NN doesnt lead to accuracy loss

 - In very high-dimensional spaces, PSO’s performance often deteriorates due to the “curse of dimensionality.” The search space grows, with a linear increase in the number of parameters / weights, and swarm communication becomes less effective.
    => When the number of dimensions increases, the search space grows exponentially, and good solutions become sparse, making optimization extremely hard.
 - Hypothesises that PSO performs poorly on large NNs due to hidden unit saturation:
 - Saturation means => The neuron outputs values very close to the extremes of its activation function.
    - like on sigmoid a neuron constanly outputs 1 ... 
 - if multiple training patterns cause hidden units to output the same values, differentiation (between classes) becomes impossible
 - If the weights cause the resulting net input signal to always be a large positive or negative number, the
hidden unit will always output a value close to either end of the activation function range.
    => Reducing hidden units to this binary output state (two ends of the activation functions output) damages the overall information capacity of the NN, causing learning to be slow and inefficient
 - This doesnt happen in GD / Backprop:
    - sees outputs are saturated, computes gradient
    - adjusts weights to bring neurons back into useful range => pushes weights to correct scale
 - Fix: initialising weights in a small interval:
    - instead of:   w ∈ [-1, 1]
    - use:          w ∈ [-0.1, 0.1]
    - That will help because constraining the PSO to a small interval around zero is hypothesised to decrease the hidden unit saturation by producing a smaller net input signal
    - z = w⋅x + b , descreasing w will not saturate z (it wont be too large), which is the input of the activation function
    - PSO_0.5 => weights are constrained to [-0.5, +0.5] at all times (not only at init)
        - constraining of the weights should help avoid saturation
        - is essentially weight clamping
 - Dropout Layer can be problematic with PSO. Iassumes the fitness evaluation is reasonably stable (same weights same fitness no randomness):
same weights → similar fitness.

## Regularization: ============================================================================
 
- reqularization penalty isnt included in a forward pass: probs = model.output(X, false); (this is inference mode)
        => would be added only if using DL4J's internal training loop.
        => need to add it explicitly
 - Another way: Introducing randomness in the training process (Dropout) / randomness-based regularization?
    - During training, randomly disables neurons. Is problematic on PSO because it needs to be stable / no random evaluations

## CNNs - PSO: =========================================================================

For CNNs especially, but also generally speaking for NNs, a hybrid is used between PSO and Gradient Descent:
 - 1) PSO => NN Architecture. This is part of NAS (Neural Architecture Search), a method used to programmatically determining NN architectures
    - similarily with hyperparameter search (hyperparameters for Adam) 
 - 2) Use something like this for direct training - PSO used for fine-tunning the weights:
    - use GD to get into a good region
    - use PSO-hybrid to escape stagnation / local traps with small data => PSO affects weights but after they are already in a good place 
    - then use GD again to refine
 - 3) Use PSO to optimize only a small subset of parameters (for example: last fully connected layer)

CNNs cant be trained by PSO:
 - CNNs have structured parameters => they are not independ like the parameters of FNNs
 - spatial correlation
 - the patterns of the filters they must learn are highly correlated. This means hierarchical learning of filters
    - Neural networks learn low-frequency (simple - edges) patterns first, then high-frequency ones (complicated pattern - objects)
    - the CNNs purpose is to extract the high-frequency pattern / feature. PSO is stack at low frequency (which is by itself meaningless)
 - Cant add batchNormilization, which is really helpful on GD
 - generally speaking works better with more layers, which PSO doesnt like 

CNNs are helped by a Dense Layer in the end:
 - CNN part: feature extractor, Dense Layer => classifier using the features of the CNN
 - Dense layer expresses interactions between features, more expressive classification boundary
 - gives more linear influence on ouput
 - CNN without Dense head is usually underpowered for classification
 - For PSO specifically, Dense layers are the easiest part to optimize

## ===================================================================================
## Theory / PSO Paramaters ===========================================================

However, major disadvantages of BP are its convergence rate is relatively slow and always being trapped at the local minima.

## Convergence vs Exploration: =========================================================

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

## Premature congvergence: =========================================================

 - trapped in a local optimum if the search environment is complex with numerous local solutions.
 - Evolutionary operators such as selection, crossover, and mutation have been introduced to the PSO to increase the diversity of the population, and to
improve the ability to escape local minima
 - collision-avoiding mechanisms to prevent particles from moving too close to each other
 - Of course, it is hoped in practice that Pm (gBest and pBest) dont remain fixed. A key source of variation is the updating of Pm over time as new points are found in the search space which are better than those previous ones. If this Pm doesnt manage to change as we go towards it => convergence, but it might be premature 

## What to look at for training process: =========================================

 - convergence (the ideal result is located, but the swarm doesnt converge on it):
    - We need to always be converging when the execution ends. If we dont then the FedAvg will yield a bad result
    - Definition:
        - Velocity collapse ||v_i|| → 0
        - Swarm collapse Var(x_i) → 0 (x_i current position of particle i, each particle collapses to the same position) => all positions become (almost) the same (no exploration). Variance for mathematic reasons

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
 - Model:
    => Is the model good enough to fit the dataset or not
    => dimensionality trade-off: high dimensionality model typically does better on gd (at least what overfitting is concerend), but this isnt the case for PSO. There dimensionality actively harms PSOs ability to optimize. 
        => trade-off between models capacity to fit and PSO capability to train the model

 - **Increasing N_WORKERS:**

    - Increasing N_WORKERS adds compute cost and may proove detrimental, for FULLY INFORMED especially
    - At the same time, N_WORKERS can help expanding the search space (this is more begenficial for neighborhood best), exploration increases.
    - As N_WORKERS increases, number of  data (batches) decreases per worker. This means: number of updates decreases, which means worse less reliable / convergence and number of times reporting current weights (for monitoring) decreases. 
        => As long as INDEPENDENT_WORKER_DATA_PROCESSING=false => count_updates decreases
    - In practice, increasing N_WORKERS is a net positive (both on time and accuracy), as long as:
        - Data per worker doesnt get reduced (happens if data is already plentiful and convergence happens already before data runs out)
            - Stable number of updates
            - Still using INDEPENDENT_WORKER_DATA_PROCESSING=false
        - Using neighborhoods (so as to not dialute the direction)
        - You can ensure true parallelism between the workers or at least this isnt computationaly too heavy

## Initialization of the population ======================================

 - Particles initial position is very important, particles need to be scattered in intialiazation
    => this applies to both weights (position) and vectors. The random generation should be seeded by particleId
    => model.init() is where the weights get randomized, and it happens internally inside DL4J/ND4J
        => Without specifying weightInit(...), DL4J uses its default weight initialization => internally decided "Xavier", "He", normal/uniform
        => Xavier is already “small-ish”, and for many layers its typical magnitudes are around 0.05–0.15
            => Xavier has a scale around 0.1–0.2 magnitude (depends on the layer)
            => U(-a, a) , where a is: sqrt(6 / (fan_in + fan_out)) # fan_in = nIn, number of Inputs in the Layer
    => randomizeVelocity(int workerId, float sigma) is where the velocities gets randomized
 - Standard PSO the particles are initialized randomly using the uniform distribution which is
    not considered as the best choice
 - If your swarm starts spread out (“scattered”) / diversity at initialization, then some particles will start in terrible places (high loss / “worst fitness”) and some will start in decent places. This is good for exploration and sampling the search space properly.
 - We want high discrepancy (a measure of how unevenly points cover a space):
    - High discrepancy: points cluster, leave gaps, oversample some areas.
    - Low discrepancy: points are spread “evenly” with minimal clustering and minimal holes 
        => classical uniform distribution
        => Sobol, Halton, Faure sequences: they fill the space more uniformly than standard RNG.
 - Bad Initial Fitnesses isnt bad, mediocere solutions are bad => If the whole swarm starts in a mediocre region, 
    they can prematurely collapse / converge and stop exploring. Especially at the start we favor exploration

 - Weight initialization Method:
    ```java
  	.seed(123 + workerId)
	.weightInit(WeightInit.XAVIER)	// or .weightInit(WeightInit.RELU) but seems to be worse scale for the weights
    ```
    - We seed with workerId for predictability and to differentiate between every worker (different positions at init)
    - WeightInit.XAVIER seems to be giving a good scale
        - Scale of the biases might be higher than the scale of normal weights
        - Is uniform based NOT uniform(−1,1), but uniform(−a,a). a is the scale and it is calculated based on layer size 
            => scale is different between every Layer
    - WeightInit.RELU:
        - slightly larger scale
    - .weightInit(WeightInit.UNIFORM)   => [−0.05,0.05] / or more custom: .weightInit(new UniformDistribution(-1.0, 1.0))
 - Caveat:
    - Biases to neurons (both FNNs and CNNs have biases) are not randomized but always initialized to 0.
        - Number of outputs == Number of Neurons 
        - standard practice  
            - symmetry breaking handled by weights (must break symmetry at initialization)
                Symmetry: 
                - two or more neurons behave identically / have the same weights, so the model can’t learn different features.
                - In GD: both neurons receive the exact same gradient, so they update in exactly the same way
                - Since weights are already random, neurons are already different / produce different gradients, symmetry is already broken
                    => no need to worry about random initinalization of biases they are attached to the already assymetrical weights
                - PSO itself breaks symmetry through particle and velocity randomization
            - avoids introducing bias before learning (no prior preference for activation)
            
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

    // some original versions dont have the Inertia term all together. Observed a significant performance decrease when doing so. 
    // FI-PSO is not meant to drop inertia entirely, by itself, FI’s social term is either:
        // too small (means it has prematurely convergenced) or too noisy
    float velocity = C1 * r1 * (pbest[k] - x_i[k]) + C2 * r2 * (gbest[k] - x_i[k]); // 3 accelarations
    ```

 - ## Clamping Velocity: ===================================================================================

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

 - # Randomness Dimensionality (CLPSO - Page 2): =================================================== 

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
    - In practice, (1) performs much better

## ==============================================================================================
## Local Version of PSO => Neighborhood based (Communication Topology) - Theory:

 - Increasing neighborhood size deteriorates performance, the worst of FIPS is on ALL topologies:    
    - The swarm behaves like a single mass + it becomes more prone to local minima => exploration decreases
    - Too many pBests leads to direction being dilouted and the particles wont move coherently. Not necessarily because of pure random behavior, but simple iability for spatial convergence. There are too many directions to be considered
        - the simultaneous influence of all the particles in the swarm ‘confounds’ the particle
        - “confound” means: to confuse / overwhelm

    - **This effect gets worse as population size increases. Not Scalable** => The paper used 40 particles 

 - Neighborhood size controls the balance between:
    - Exploitation (large neighborhoods)
    - Exploration (small neighborhoods - decreased neighborhood size):
        - With decreased neighborhood size / scope of a particle, there is a delay in the information spread through the graph.
        - Less convergence, more randmoness exploration
        - Less communication / Communication Inhibition

 - Particles that are acquainted to one another (in the same Neighborhood) tend to explore the same region of the search space (lower level search). Neighborhood topology affects also relationships between neighborhood (higher level search)

  ## Protocol:
    - Particles get information only from their own neighborhoods best => local_best instead of gBest.
    - Communicatio Inhibition / Delay:
        - If individual i finds a good solution, this may be passed toits adjacent neighbor j, but not immediately to k, which is not connected to i.
    - Neighbors == Topological Neighbors (doesnt change during a run)
        - v_i(t + 1) =  w * v_i(t) + c1 * r1 * (pbest - X) + c2 * r2 * (lbest - X)
        - Neighborhood PSO does not mean neighborhoods are disjoint clusters, they are cirularly dependent.
    - **Ring topology** (a cycle graph) - URing - Most commonly used:
        - The population is arranged in a ring (particles == nodes in a ring), for example in 40 particles:
            - 0 — 1 — 2 — 3 — 4 — 5 — ... — 39 — back to 0
        - a neighborhood of six, or three topological neighbors on each side. means that particle_i has:
            - i-3, i-2, i-1, i+1, i+2, i+3 as neighbors 
            - Topologically, every particle has its own neighborhood and neighborhoods overlap heavily
            - neighbors(i) = {i-3, i-2, i-1, i+1, i+2, i+3} mod P   # P == number of particles, this is a circle. Its length is the global parameter neighborhood size 
    - **Square Topology** => every particle is connected with 4 others. Particles can be aranged in a 2D grid and the neighbors are the 4 closest particles (both on the horizontal and vertical axes)

    - If particle i finds new pbest then only particles whose neighborhoods include i might update their lBest:
        => this is because different particles always see different neighborhoods
        => lBest_k = the best pBest among neighbors(k)
        => Examples:    1) Particle 10 improves pBest, articles that may update 7, 8, 9, 11, 12, 13
                        2) particle 9 => neighbors(9) = {6,7,8,10,11,12} recomputes lBest_9 = best( pBest_6, pBest_7, pBest_8, pBest_10, pBest_11, pBest_12 )

 - Number of Neighborhoods == 15% of the number of particles
    - 40 particles => 40 * 0.15 = 6 Neighborhood size + Number of neighborhoods = 40 (once per particle)

 - PSO with a small neighborhood might perform better on complex problems, 
   while PSO with a large neighborhood would perform better on simple problems
    - Point of neighborhood PSO is that global Best (converges faster, but might collapse on a local minima - minimize the loss function) vs lBest (less premature convergence - keeps diversity longer)
 - Topology Comparison:
    - All: 
        - Basically disabling Neighborhoods. K (neighborhood size) == N (number of particles)
    - Ring:
        - Most reliable / stable Neighborhood topology is URing, which never failed when implemented with the wFIPS algorithm.
        - Without the self (the same particle doesnt include it self in the prefix, self is removed from the neighborhood topology) => "U" prefix
        - This goes contrary to particle swarm lore, which describes the algorithm in terms of the combination of “cognitive” and “social” experience.
            => in URing the "self" is missing
        - Self (protocol) => the owns pBest influence is more important, your own pBest gets 0.5 * c, while if N = 4, others gets the other half => 0.5 * c / 4
            => own pBest gets half of c (φmax), all the others get the other half of c 
            => keeps diversity longer, FIPS may coverge too fast
        - Ring has slower spread of information, it has higher average distance (average number of edges between two nodes in neighborhood graph)
            - Generally speaking it requires more time to converge
        - Expected Neighborhoodsize: {4, 6, 8}
    - Square/von-Neumann Topology spreads faster 
        - Information flow speed (like how quickly a pBest spreads) is higher, Information spreads between the particles faster
        - Neighborhoodsize: 4
        - good in-between Ring vs All 
    - End Note: The best performance of all occurred in the selfless-square FIPS configuration. Selfless is good, because we are actually either way considering the self because of X (Pm - X, aka current position habbit)

- gBest vs pBest:
    - In gBest, neighborhood size means how many other particles you can choose among, and the more there are, the better the one you pick is likely to be. 
    - In the fully informed neighborhood, however, all neighbors are a source of influence. Thus,
    neighborhood size determines how diverse your influences will be. This might be detrimental (search becomes detrimental).
        

## Neighborhood on Kafka =========================================================

 - Will not work in a Kafka / Kafka Streams setting, because:
 - shouldnt implement it the way it was designed it costs too much => Coordinator needs to send way more gBest (now lBest) messages, especially as the number of particles increases. Each particle would need to receive lBest messages just to reject them based on their ID.
 - Possible workarounds is with Keyed-by-neighborhood / keyed-by-particle (routing via partitions):
    - Make lBest a per-particle stream/table and route only relevant updates.
    
    Protocol 1) (Like it because of Router) - Router Fanout (x6):
    The protocol (still has significant mesaage overhead based on neighborhood size):
        Particle i publishes pBest to Broker / Topic: pbest-updates with Key i
        Router (new Kafka Streams app / the coordinator) consumes pbest-updates and republishes to neighbors
            Router computes neighbors N(i) = {i-3,i-2,i-1,i+1,i+2,i+3} mod P and emits 6 new records via the neighbor-candidates topic, Key = k (the neighbor particle id)
                => neighbor-candidates has P partitions (number of particles)
        Without a router:
            particle must publish 6 records every time, to neighbor-candidates (one per neighbor key)   
        Each particle k consumes only its own neighbor candidates and computes lBest_k = argmin fitness among received neighbor pbests

    Protocol 2) (Less overhead):
    Each particle publishes pBest to PBEST_WEIGHTS_TOPIC
    Coordinator keeps all pBest_i and computes lBest_i = argmin{ fitness(pBest_j) | j in neighbors(i) ∪ {i} }
    Coordinator publishes lBest_i updates to a single topic LBEST_TOPIC keyed by i (P messages / each particle is an individual lbest message), Each particle consumes only its own key i.
 
    Protocol 3) Keep Kafka Topology, just filter based on Neighborhood + WorkerId

More problems:
 - Partition count of a broker is stable   
 - Increasing partitions (manually) is allowed, but it changes (key, partition) mapping for new partitions
 - Cant use “one partition per worker” as core design, since we need elastic workers (controlled via N_WORKERS)

The goal is to reduce the total amount of messages being sent and received 
a worker might with N_WORKERS = 40 receive 40 messages only to drop the 35 ...

the solution you are proposing is hard but seems to solve the problem but it only creates a bigger on ... now the Router needs to send a bunch of messages, lets say Neighborhood size = 6. It needs to sends 6 for each different worker (we will have 1 partition per workerisnt this what you are proposing ?) thats sending 6 * 40 messages ...

Current (GlobalKTable “broadcast-like”)
    Each pBest update is seen by every worker:
    Produced: N
    Consumed (deliveries): N * N = N² (N Workers consume N messages. Holistically)
N = 40 => 1640 deliveries
N = 10 => 110 deliveries

Router fanout (targeted)
    Each pBest update is forwarded only to the workers that need it. K == Neighborhood size = 6
    Produced by workers: N
    Produced by router: N * K
    Consumed by workers: N * K
N = 40 => 520 deliveries
N = 10 => 130 deliveries
! Problem: Kafka doesn’t “know workerId”; it assigns partitions to consumers:
    - Kafka Streams uses consumer group assignment / protocol → partitions go to whoever is alive, i.e. StreamsPartitionAssignor is used.
    - You cannot do consumer.assign() (manual assignment) inside Kafka Streams. You can in a plain Kafka Consumer. This means a worker cant decide which partitions to subscribe to.
    - You can choose to which partition to send to (by keyes) but not which Kafka Streams instance receives from what partition / gets assigned to what partition

Worker fanout (targeted) - Without the router, just have worker make K-duplicates 
    Produced by workers: N * K
    Consumed by workers: N * K
    => this saves on some stuff, but makes workers do more work + Router better for diplomatiki

Potential Solution => Create a variable amount of topics, one per worker ... (insane)

## Fully Informed ===================================================================

 - In classic PSO, in neighborhood best, there is no assumption, that the best neighbor at time actually
found a better region than the second or third best neighbors (they may not have gone deep enough to their corresponding regions yet).
 - This way, important information about the search space may be neglected through overemphasis on the single best neighbor. 
 - New Formula, because all the neighbors contribute to the velocity adjustment, we say that the particle is fully informed.
 - 𝜑_k = φmax​/∣N∣, This is to keep the total expected acceleration roughly constant as neighborhood size changes.
    - N = number of neighbors, φ_k will the same coefficient applied equally to all neighborhoods
    - generally speaking you need to make the contribution of the social coefficient independent from number of particles and from N_WORKERS
    - might have improved handling of increased amount of workers
 - In the abscence of improvement (of the neighborhood best particles), aka in stagnation, the swarm converges toward the centroid of its neighbors’ previous best positions.
    - This creates a built in bias towards the centroid of the topology / neighborhood. 
        - Loss Landscape: single funnels or multiple funnels (funnels => like a physical bowl, its going down (its a low not a peak), because we are trying to minimize loss). Many different starting points all “flow” downhill toward the same good solution
    - So even if one neighbor has a great pBest far away, the combined pull from everyone tends to drag you toward the middle of the group.
        - If this bias is good or not depends from the optimization landscape (of the function)
    - This effected increases with population size, increasing N_WORKERS generally should help exploration, but in this case, due to the increased central pull, its harming it
    - This is why with increased neighborhood size we get stronger smaller exploration

## ================================================================================================
## Communication: =================================================================================

Kafka / Kafka Streams => this is a non centralized enviroment. This is why these should be considered 

    - End-to-end propagation delay => how quickly a particle / worker becomes up to date. This means that, during the delay, the particle will be computing with old information about pBest / gBest (which doesnt happen in centralized)
        => this essentially slows down convergence, but may improve exploration
        => this isnt necessarily something bad, but it is something that we need to adjust for parametrically in comparison to a centralized PSO enviroment

    - CPU time spent in communication related work:
        - Time spent serializing/deserializing, copying bytes, updating the store, etc.
        - this is overhead time

    - Independent of time, how many messages / bytes where sent. This may impact energy consumed, which may matter depending on the enviroment 

### Filtering / Communication prevention: =========================================================

    - This is an asynchronous protocol. We dont have the ability to stop pipeline / incoming data from Kafka topics
    - But we have the ability for workers / coordinator to choose not to send messages, when it is predicted to be unnecessary 
    
    - Filtering effects overhead and sometimes delay
        => Delay will be affected only in case of congsetion    

    - Given the Kafka Topology, we can influence only the communication pipelines 2_1, 2_2 and 6:
        - A) 2_2 is the federated learning pipeline. It is used to extract the average model, but this doesnt actively partiticipate in PSO.
            - This should be selected as the users preference, since through this form of communication the progression of the algorithm is reported (CLI)

        - B) 2_1 and 6 are directly used for PSO (pBest / gBest weight messages). 
            - We can choose not to send them if the loss of the new model wasnt significantly improved 

    A)
     - Just do a * 4 on N_BATCHES (this happens in config)
     - OR just increase the monitoring / sampling rate as the swarm converges to identify the best solution (because at the start, as the swarm isnt converged, the accueracy / model performance is as good as random since the particles are far away and their average is wherever)

    B) Filter based on:

    - Significant Loss Threshold:
        - Decrease over time (strict → permissive)  
        - Increasing over time would mean you demand bigger improvements later — but later improvements are inherently smaller. That’s a recipe for “everyone stops talking” and accuracy tanks.
        - We want exploration first then exploitation
        - Using threshold function: θ(t)=θmin​+(θmax​−θmin​)e^(−t/τ)
        - This only approaches θmin asymptotically as t → ∞.
        - Need to determine τ (TAU) using θ(MAX_COUNT) = θmin​+ε and τ = MAX_COUNT / ln((θmax​−θmin​)/ε)
    
    - Batch pBest Messages / Records not accumulatitevely, but replacingly ....
        - i need to take advantage of this batching policy stuff to improve my filtering ... this is only possible if i am able to, instead of accumulationg records ion a batch, to replace them / update them before sending the batch ... so the the actual batch size would be 1 but you eould get the benefits from batching => for my app this makes sense ... there is no rason to send 3 pBest records at the same time i noly keep the most recent one:
        - set props.put(StreamsConfig.producerPrefix(ProducerConfig.LINGER_MS_CONFIG), 0); but it doesnt really matter, because it is enforced already by PBEST_DEBOUNCE_MS
            => LINGERING should be set to 0 either way. In every case sending two weight messages at the same time serves no purpose. Only the latest / most recent one is going to be used. 
        - can be implemented also in a Kafka Streams way

    ## Filtering Ideas ======================================

     - Send only if its a significant distance away from the previous pBest => maybe you can combine it with the significant loss 
        => Problems: this would have an effect only in the case of convergence, but in this case we actually want to converge to the best possible solution
        => This would harm convergence when convergence is needed most
        => minute changes in the distance can significantly change loss, which is what we care about, especially during convergence
        => In short, we would a combination of the two (send only if there is significant distance, but the threshold for significant distance decreases over time)
            => But we dont really care if there is a significant distance in the first place, we know there is a distance already 
                => The opposite could prove more effective => send only once per 50 * N_BATCHES (just to check progress) or if distance is small
                
    - Send pBest only when it beats a reference quality gate:
        - I may only not send based on loss using a previous reference. Using a flat loss cutoff is problematic, loss / accuracy is NUM_CLASS dependend and for some cases learning happens rather slowly with small adjustments. We are evaluating based on significant relative improvement.
            => we need to slowly warm up to a correct solution we cant reject it because its not good enough yet 

    - If there is congestion be more strict:
        => these is no congestion
    
    - Suppress updates that are worse than what other neighbors already have.
        => Fully Informed losses its meaning or at the very worst we are stuck with old pBest vaules

## ==================================================================================
## Functional Requirements: =========================================================


## Non Functional Requirements: =========================================================

	- θελουμε καλο accuracy γρηγορα (trade off) δηλαδη τα δεδομενα πρεπει να επεξεργαζονται γρηγορα για να ειναι streaming περιβαλλον

## ========================================================================================
## Experimentation & Performance: =========================================================

 - Load 400000 messages / samples to Kafka Input topic (make the reperation number just high enough for this)
    - divide this by 40 partitions => each partition gets 10000 samples

 - Algorithm effieciency / performance should be measured as the Kafka Record processing time (of WorkerTransformer) of a batch:
    - What we aim to is the forwardPass to be the primary source of delay (the rest of the processing shouldnt cost more that + 4 ms)
    - We cant improve the forward Pass. Its complicated and handled by DL4J, except to:
        - Activate GPU => adds too much overhead if Neural Network is simple
        - Make Neural Network simpler, which costs on Neural Network quality / fitting potency

 - Need to measure:
    - Diagramm 1: y-axis: Accuracy - N_Workers
    - Diagramm 2: y-axis: Training Time - N_Workers

 - Run for N_Workers = [5, 10, 15, 20]
 
 - Performance Measurement Methods:
    - htop (shows logical CPUs and ||| represent CPU time usage on each logical CPU)
        - 6 physical cores × 2 threads = 12 logical CPUs
        - 12 physical cores × 1 thread = 12 logical CPUs
        - use <code>lscpu</code> to evaluate CPU number and number of threads
            - number of physical cores: Core(s) per socket: 6
            - number of threads on its core: Thread(s) per core: 2

 - Performance Metrics:
    - accuracy - N_WORKERS
    - time - N_WORKERS 
    - accuracy after fixed time given / data given 
    - time / data needed to reach accuracy goal    

 - Kafka performance bottleneck:
    - Increasing the number of consumers per partition (identified from the tuple (application, task)) on a single broker will lead to the inability of the broker to handle all of them, and it will drop a frew
        - broker too busy => missed heartbeats and polls => then drops the tasks
    - If tasks within an application remain idle for too long, then they miss, as a Kafka Consumer, heartbeats / they dont poll fast enough, then Broker rebalances and kills the task.
        - consumer doesn’t call poll() (asks for new data) often enough → exceeds max.poll.interval.ms
        - slow polling, happens because of heavy processing + many independent apps
        - Rebalancing == the ownership of partitions changes
        - This causes rebalancing, which causes task churn
        - Task churn means that Kafka Streams keeps creating tasks, then revoking/closing them, then creating them again, repeatedly. rapid task create/close cycles, often with 0 records processed in between.
    
    - Idle for too long happens because of the computers capacity for parallelism (its limited):
        - If some tasks take too long / overall processing takes too long, then some tasks get starved
        - Starved tasks => dont communicate with broker, get kicked out 

    - ## GPU: ================================================================================

        The GPU is capable of executing CNN forward pass faster than the CPU. 
        GPU is only used for DL4J stuff (wherever we are handling DL4J):
            - INDArray probs = model.output(X, false);    
            1) [CPU parsing + copying]
            2) [CPU → GPU transfer] (Copies data from CPU RAM → GPU VRAM, this is a memory transfer)
                => Memory copy overhead
                => Happens on: probs = model.output(X, false); (transfer X) - This is BOTH Memory transfer + forward pass
                => read/copy from Java heap → ND4J buffer (still host-side initially), copy X host → device (CPU RAM → GPU VRAM) if it isn’t already there
            3) [GPU forward pass]
            4) [GPU → CPU sync]     (Pull data back to CPU, synchronizing on every batch => get results when they are needed)
                => normally .output() would be async but in this case we need the results immidiatly to calculate loss
                => Sync means => CPU is blocked until the results from the GPU have arrived        
                => is enforced from commands like these which request probs / the result:  float[] flatProps = probs.data().asFloat();  

            5) [CPU loss + accuracy loops]  

        ## GPU Overhead:

         - (2) + (4) are overhead (+ GPU scheduling / Kernel launch). If the forward pass cost is small either way, then its not worth it to use GPU, it will end up costing more time. This happens specifically on the Dense NNs where CPU is prefered. For CNNs, gpu is confirmed.
         - Competition between N_WORKERS for the GPU. Another source of overhead are the N_WORKERS: 

            ## Time:
            - They need to share the GPU. There are 6 CPU cores working in paralleland the GPU is only device (you would like N_WORKERS == N_DEVICES). GPU has Kernel launch queue, it can only launch a limited amount of Kernels. Many workers fight over the GPU, since every time its used one time.
            - GPU can parallelize compute internally => The highway = thousands of parallel GPU stuff
            - The toll gate = kernel launch + memory transfer + sync => is triggered at the model.output => N_WORKER competition overhead is included in the forward pass.

            ## Memory:
            - Whatever has to do with DL4J can live in the GPU
            - If too many memory allocations happen between many N_WORKERS, then GPU will not have the time to clean (free) the memory each time (there is a delayed release if excplicitly freed). The allocating memory rate will become bigger than the cleaning memory rate as N_WORKERS increases (leading to a crash, because of Memory overflow).
                - Those arrays might still be referenced, this is why they arent getting cleaned

            - **Real Memory Expense:** On .output (forward pass), GPU needs to:
                - allocate activation / intermediate tensors / NDArray => every intermediate / hidden layer each produces intermediate data. Expensive are:
                    - depthwise conv outputs, batchnorm / activation outputs, ... (just hidden layers of the model)

                - activations are the output tensors of a layer (they are the transformed versions of the input for each layer):
                    => input: (batch, 32, 32, 3) => Layer: Conv2D(32) => (batch, 32, 32, 32)
                    => This is why batch size matters for memory. This entire activation (dimensionality wise) needs to be saved for each layer
                    => During training, we must keep activations because of backpropagation (to calculate gradients you need the forward outputs of each layer)
                        => stored during forward pass, reused during backward pass
                    => So real VRAM use may be closer to 2 × activations, because of gradients for each activation and other temporary buffers
                    => For CNNs: Activations dominate memory, not parameters
                        => MobileNetV2 has only ~14MB weights, but during training it may require hundreds of MB or even GB VRAM
                    => Lifetime of Activations: After forward pass + backward pass → they are released / overwritten
                        => Training: intermediates must be kept for backward pass → big persistent “activation stash”
                        => Inference: intermediates are temporary, they can be reused/freed immediately after each layer / each forward pass 
                    => Freezing layers should stop memory being spent on activations for backpropagation (it doesnt)
                        => gradients are not computed/propagated for them ⇒ you shouldn’t need to retain their intermediate activations for backward.

            - Memory floor: 
                There is a baseline VRAM “floor” that doesn’t go away:
                 - CUDA context + cuDNN handles, cuDNN convolution workspaces (often big), the memory might get reserved/cached, and just not evicted because VRAM not full (ND4J/CUDA caching allocator / memory pool (keeps memory reserved for speed)), model parameters / NDArrays resident on device (and possibly extra buffers)
                 -loads libraries, allocate internal state and caches => This memory stays allocated until the process exits. The cuda code is loaded in the GPU

            ## NVIDIA cuDNN (DNN = Deep Neural Network):

            CUDA Deep Neural Network library
            - cuDNN convolution algorithms often require a “workspace” scratch buffer.
                => use .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE) to try and use smallest amount of memory
                => highly optimized GPU library
            - DL4J models this by giving certain layers an internal field commonly named helper.
            - If a layer can use an accelerated backend implementation (like cuDNN), DL4J will create a helper object for it.

            While training, parameters + optimizer (Adam / PSO (velocity)) state + activations live in memory (RAM / VRAM). If using GPU, the variables/weights are usually placed on the GPU (VRAM) so computation stays on-device.
            - model.params() points to CUDA memory
            - setData(float[]) uploads to GPU
            - INDArray is “CUDA-capable” => means it can be bound / copied to the gpu

            - Model size memory consumption:
                - float32 = 4 bytes/param
                - Size ≈ 288,298 × 4 = 1,153,192 bytes ≈ 1.15 MB
                - this also gets a * 4 because of other parametes like gradients, Adam stuff => 4 × 1.10 MiB = ~4.4 MiB
                - This is negligable to the memory consumed by the activations (intermediate data / feature maps)


        Memory Phenomenon:
            - Memory Leak: Memory is never freed => some GPU arrays stay referenced (pointer) and never get released. The garbage collector cant free them
            - ND4J uses a caching allocator on GPU: allocator growth / caching

        The forward pass cost is also determined by batch size, but this needs to be kept small for PSO not to run out of data.
            => On a simple NN, cpu is preferable
            => GPU is worth it if: ForwardPassTime >> Transfer + Sync cost

        But GPU is limited, it will only help in neural net compute.
            - No contribution for serialization / Kafka messages
            - This means me may be able to afford bigger models or batches, but the primary bottleneck will still be Kafka / CPU Scheduling
            - CUDA just does faster tensor math, convolutions, matrix multiplications

        GPU gets more benefit from increased batch size
        
        
# DL4J Memory Management: ==========================================================

DL4J has 3 different memory spaces:

 - CPU RAM (both ON-HEAP and OFF-HEAP are on CPU RAM):
    - JVM Heap ON-HEAP (Java Memory - still technically native memory)
        => Controlled by    - Xms (how much memory does JVM Heap get at start) 
                            - Xmx (JVM Heap memory limit)
        => Not used for INDArray tensors
        => on heap - off heap matters only for CPU RAM

    - ND4J OFF-HEAP (native CPU memory) - managed by DL4J not JVM - JavaCPP allocations:
        => Controlled by -Dorg.bytedeco.javacpp.maxbytes
        => Used mainly for INDArray tensors
    
 - GPU VRAM:   
        => Controlled by -Dorg.bytedeco.javacpp.maxbytes
            - ND4J off-heap size ≈ GPU memory usable 
            - The underlying ND4J arrays are off-heap, and with the CUDA backend they are (effectively) backed by GPU memory for GPU execution.
        => ND4J mirrors OFF-HEAP buffers to GPU
            - This means that on CPU => GPU communication, NDArray Buffers are exchanged off heap (copied from CPU off-heap to GPU. If CPU off-heap is limited, then GPU VRAM is limited in the same way) => ff-heap allocations are “mapped” to GPU memory
        => ND4J CUDA uses JavaCPP (bytedeco) to allocate native memory and manage CUDA resources.
            - JavaCPP (bytedeco) is the bridge between Java and native code (code of the CPU)
            - this is generally necessary when not on JVM / on Heap. The RAM is managed natively by C.
            - JavaCPP will try to keep native allocations it tracks under this budget (mostly host /off-heap), but CUDA/ND4J can still reserve/hold VRAM via its own pools/caches and via CUDA/cuDNN (this is what is reported by nvidia-smi).
        => It is possible to use HOST-only memory with a CUDA backend, but its not recommended for performance

 - Reasons why limiter doesnt work and it will keep on allocating:
    - You cant limit what the GPU is doing beyond the tensor allocations:
        - Dorg.bytedeco.javacpp.maxbytes ONLY limits ND4J-managed tensor memory pools (the stuff backing your INDArrays and workspaces)
        - All the other GPU memory (CUDA context, cuDNN (fastest algorithm - takes up a large buffer. For a huge model like MobileNetV2, this might be up to 500MB), convolution workspaces, kernels, caching allocator, etc.) is NOT limited by that flag.
             => these memory allocations are used for the actuall convolution, not the memory transfer
    - You cant control CUDA / GPU caching 

 - 1)  -Dorg.bytedeco.javacpp.maxbytes => limits JavaCPP’s own tracked allocations , off-heap host memory
            - This INDIRECTLY effects memory usage if tensor size / transfer is the bottleneck

 - 2) -Dorg.bytedeco.javacpp.maxphysicalbytes => total physical memory footprint of the process
        => both on heap and off heap => is set by default to maxphysicalbytes = maxbytes + Xmx + extra
        => also influnces GPU allocations to a greater degree

# htop Alternatives for GPU: ==========================================================

```bash
nvidia-smi	# Confirm if it was successfully installed
watch -n 0.5 nvidia-smi
watch -n 1 -t nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
```

 - Performance Theory:
    - True parallelism comes from CPU cores
    - Swap = “RAM overflow to disk” (very slow), if Swap and RAM is huge => problem

 - Kill Zombie Java processes:
```bash
ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -n 25   # detect them
sudo pkill -2 java
sudo pkill -9 -f java
```

# We see that docker writes inside /tmp/kafka-logs, which is inside the container not the host. Docker keeps the container filesystem, doesnt delete it.
```bash
docker exec -it broker sh -lc       # runs it as a shell inside the docker container   
docker exec -it broker sh -lc 'du -sh /tmp/kafka-logs'
docker exec -it broker sh -lc 'du -sh /tmp/kafka-logs/*'  # show per partition
```

## =========================================================================
## Extra Sources:

 - https://www.quora.com/Is-particle-swarm-optimization-an-appropriate-way-to-train-a-deep-convolutional-neural-network-for-image-recognition
 - https://spotintelligence.com/2025/10/20/particle-swarm-optimization-pso/



## Transfer Learning ==============================================================

Ranked from simplest to heaviest:
 - Tier 0:
    LesNet
    MobileNetV3Small
    MobileNetV2
    EfficientNetB0
    NASNetMobile

 - Tier 1:
    MobileNetV3Large
    EfficientNetB1
    ResNet50 (or ResNet50V2)

 - Tier 2:
    ResNet101
    InceptionV3
    Xception
    DenseNet121

 - Tier 3: heavy 
    - DenseNet169 / DenseNet201
    - EfficientNetB2 / B3
    - InceptionResNetV2

 - Tier 4: don’t run on a laptop” 
    - VGG16 / VGG19 (huge activations + tons of parameters; also slow)
    - EfficientNetB4–B7
    - NASNetLarge
    - ResNet152

 - For DL4J:
 LeNet → SimpleCNN → TextGenerationLSTM → FaceNetNN4Small2 → Darknet19 → TinyYOLO → AlexNet → VGG16 → VGG19 → ResNet50 → InceptionResNetV1
 
 - use GlobalAveragePooling2D() instead of Flatten + Dense. Flatten costs a lot ...
 - MobileNetV2/V3 typically require at least ~32×32 (often more depending on implementation). 28×28 can fail or give junky shapes. This is because of the DownSample Layers (MaxPooling)



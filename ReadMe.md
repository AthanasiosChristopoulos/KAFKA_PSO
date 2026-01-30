## =====================================================================================================================
## Run ============================================================================================================

```bash
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

## =====================================================================================================================
## Partitioning: =======================================================================================================

N_WORKERS < N_PARTITIONS is not a problem, because if N_PARTITIONS = 40, then:
    5 workers ⇒ each gets ~8 partitions (if 40 partitions)
    10 workers ⇒ each gets ~4 partitions
    20 workers ⇒ each gets ~2 partitions

If N_WORKERS > N_PARTITIONS, then #(N_WORKERS - N_PARTITIONS) workers will remain idle / will have 0 partitions assigned.

## Project Architecture Description:

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



## Distributed, data parallel PSO Protocol: =====================================================================================================================

1) Initialization of particles, randomize their initial positions + velocities
    => initialize each particle with the same global model architecture (the architecture never changes, only the weights)
    => assign each worker (N WORKERS, working in parallel) 1 particle (this number could vary, worker could be assigned 1...M particles)
    => distribute training data to each particle 

    While True loop (break condition inside this logic):
        2) Each worker does:
                
            Repeat this for N_TRAIN_SIZE:
                => evaluate the current position using a batch of data (TRAIN_SIZE) and a loss function (non differentiable):
                => if this is a personal best loss, update pBest (personal best weights - model).
                    => communicate also the pBest to PBEST_WEIGHTS_TOPIC, where everyone will read it

                => Receive pBests of all particles (or gBest) from Coordinator
                => update velocity (potentially using new pBests) and calculate next position x_i_1 using new velocity value
                    => <code> v_i_new = w * v_i + (c / M) * sum(j, random * (pBest_j - x_i) </code>
                    => <code> x_i_1 = x_i + v_i_1 </code>
                
            => sends the current position x_i (for FedAvg) and then return to original loop

        3) The coordinator does:
            => Coordinator receives pBest and updates them (either updates whole pBest list or just gBest), informing the workers
            => Averages x_i of all particles into x_g and use that to evaluate overall performance of the model
                => Only if this x_g has a high enough accuracy (higher than DESIRED_ACCURACY) or if the DATA_TOPIC / TEST_TOPIC data has been exhausted,
                   does training conclude.
                => The execution doesnt end, since now the global best model will be used for inference of the data in PREDICTION_INPUT_TOPIC.
          
                
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

## Improve congvergence: =========================================================

 - change model
 - change constants => velocity, inertia, C1, C2
 - increase the number of children
 - look how velocity amplitude behaves
    - velocity show always start big and then becose smaller
 -  Fully Informed seems to be slower, but converging more surely (its always improving)

## What to look at for training process: =========================================

 - convergence (the ideal result is located, but the swarm doesnt converge on it)
    - this means the velocity magnitude needs to be decreasing over time => not staying constant / or getting clamped
    - Velocity is initialized with a significant amplitude which should decrease over time since INERTIA < 1\
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

## Velocity:  =========================================================

 - Is initialized to have a significant amplitude at the start
 - Inertia parameters should be adjusted so that velocity decreases slowly overtime as swarm converges
 		- velocity like simulated annealing ? Make it reduce over time
 - **W_INERTIA_FULLY > W_INERTIA_G_BEST** because we have to make up for extra directional addition in velocity:
    ```java
    float velocity = W_INERTIA * velocity[k] + C1 * r1 * (pbest[k] - x_i[k]) + C2 * r2 * (gbest[k] - x_i[k]);
    float velocity = W_INERTIA * velocity[k] + socialAggregate[k]
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

### Covertype:

    7 Classes
    85% on Gradient Descent, 80% on PSO
    Uneven distribution:
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

### MNIST:

    - Grayscale images, very simple image dataset
        - When flattend there are only 784 features
        
    - Doesnt need a convolutional neural network, because digits are always centered and a pattern will always be at the same location

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


### Pendigits-HALF: =========================================================

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



## Non Functional Requirements: =========================================================

	- θελουμε καλο accuracy γρηγορα (trade off) δηλαδη τα δεδομενα πρεπει να επεξεργαζονται γρηγορα για να ειναι streaming περιβαλλον

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

    - nvidia-smi -l 1
    - <code>nvidia-smi -q</code>  // see gpu specs
 - Performance Theory:
    - True parallelism comes from CPU cores
    - Swap = “RAM overflow to disk” (very slow), if Swap and RAM is huge => problem

 - Kill Zombie Java processes:
    ```bash
    ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -n 25   # detect them
    sudo pkill -2 java
    ```


## Run

```bash
mvn -q -DskipTests -Dexec.mainClass=pso.Simulation clean compile exec:java
mvn -q -DskipTests -Dexec.mainClass=evaluate.EvaluateIrisModel clean compile exec:java
mvn -q -DskipTests -Dexec.mainClass=evaluate.ExportDl4jModel clean compile exec:java
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

## Distributed, data parallel PSO Protocol:

1) Initialization of particles, randomize their initial positions + velocities
    => initialize each particle with the same global model architecture (the architecture never changes, only the weights)
    => assign each worker (N WORKERS, working in parallel) 1 particle (this number could vary, worker could be assigned 1...M particles)
    => distribute training data to each particle 

    Repeat this until global model converges to an accuracy > 95%:
        2) Each worker does:
                
            Repeat this for N_TRAIN_SIZE:
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
                

## Git:

```bash

git clone https://github.com/AthanasiosChristopoulos/Kafka_PSO.git
git push https://github.com/AthanasiosChristopoulos/Kafka_PSO.git

git branch
git branch -d branch_name
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

# Notes: =========================================================================

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
gunzip SUSY.csv.gz

 - 1) Add new .env variables
 - 2) Update NEURAL_INPUT and NEURAL_OUTPUT in Config.java
 - 3) Define new createModel function in Dl4jModelFactory
 - 4) Create train and test Kafka Topics
 - 5) Append in data_producer.py the load_dataset() function an elif

 - Extra:
    - update in evaluate_model.py, by adding elifs to functions:
            - NEURAL_INPUT and NEURAL_OUTPUT
            - reconstruct_layer_weights_for_keras()
            - build_keras_model()
            - evaluate_model()

## New Dataset - Specifications:

 - Needs to be a well known ML dataset.
 - Not too hard, but not as easy as iris
 - Needs to come, not from python, but externally in like a .csv
 - The model that is going to be used on it should be between 10000 - 100000
 - i need you to find a tensorflow solution online which achieves a high accuracy
 - i need multiple classes (5 up to 10) and each class has about the same class appearance frequency 
        - even class distribution among the samples
 - no CNN (like no image recognition)
 - have enough rows / samples like somewhere near 80000
 - have a reasonable amount of features (not over 100)
 
## Improve congvergence:
 - change model
 - change constants => velocity, inertia, C1, C2
 - increase the number of children
 - look how velocity amplitude behaves
    - velocity show always start big and then becose smaller

## What to look at for training process:
 - convergence (the ideal result is located, but the swarm doesnt converge on it)
 - the ideal result will not be located
 - The swarm converged on bad solution / local maximum
 - Trade-off between exploration and convergence

## Velocity:
 - Is initialized to have a significant amplitude at the start
 - Inertia parameters should be adjusted so that velocity decreases slowly overtime as swarm converges
 		- velocity like simulated annealing ? Make it reduce over time
 - **W_INERTIA_FULLY > W_INERTIA_G_BEST** because we have to make up for extra directional addition in velocity:
    ```java
    float velocity = W_INERTIA * velocity[k] + C1 * r1 * (pbest[k] - x_i[k]) + C2 * r2 * (gbest[k] - x_i[k]);
    float velocity = W_INERTIA * velocity[k] + socialAggregate[k]
    ```

## DATASETS: =========================================================

### Susy:
    2 Classes
    Balanced
    80% on Gradient Descent, 77% on PSO

### Bank:
    2 Classes
    Unbalanced 1/10 vs 9/10
    90% Gradient Descent, 90% on PSO
    Did the client subscribe to a bank term deposit after the marketing phone calls ?
        yes → the client did subscribe (opened a term deposit)
        no → the client did not subscribe

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
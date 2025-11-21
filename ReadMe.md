
===================================================================================
## Run

```bash
mvn -q -DskipTests -Dexec.mainClass=pso.Simulation clean compile exec:java
mvn -q -DskipTests -Dexec.mainClass=evaluate.EvaluateIrisModel clean compile exec:java
mvn -q -DskipTests -Dexec.mainClass=evaluate.ExportDl4jModel clean compile exec:java
```

===================================================================================
## Formulas for PSO / velocity update:

 - Neighbor best (classical PSO):
    - v_i(t + 1) = c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X) + w * v_i(t)

 - Fully informed:
    - v_i(t + 1) = w * v_i(t) + (c / M) * sum_{j=1..M} [ ρ_ij(t) ⊙ (pBest_j - x_i(t)) ]


===================================================================================

```bash

chmod 777 run_streams.sh
dos2unix run_streams.sh
./run_streams.sh --reset

```

===================================================================================

## To Do:

 - Implement as much as in Kafka Streams, as you can 
 - Implement Inference pipeline using x_g
 - Use loss functions instead of accuracy


## Distributed, data parallel PSO Protocol:

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
                

===================================================================================
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
	
```
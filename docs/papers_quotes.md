
## 1. Particle Swarm Optimisation for Evolving Deep Neural Networks for Image Classification by Evolving and Stacking Transferable Blocks

Therefore, the final encoded vector only consists of two dimensions, which are the growth rate and the number of layers
    - A PSO particle is not a weight vector

Apply Adam optimisation ... => 
    For each PSO particle:
    Build a CNN block
    Train its weights using Adam

The whole training set is used to train and evaluate the stacked candidate, and Adam optimisation is chosen to train the stacked CNNs
    gradient-based optimization => Adam 

That’s why every serious PSO–CNN paper does:
    PSO → architecture / hyperparameters
    Gradient methods → weights
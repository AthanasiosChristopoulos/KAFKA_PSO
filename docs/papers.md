
## ================================================================================================
## Papers I have already downloaded:

Strong sources:
IEEE
Springer
Elsevier
ACM
NeurIPS / ICML / ICLR

Medium:
University journals (like this one)

Bad: No DOI

## ====================================================================
1)
Zhang, Zhang, Lok, Lyu (2007), A hybrid particle swarm optimization–back-propagation algorithm for feedforward neural network training. => The paper explicitly says it combines PSO global search with BP local search to train network weights, and it uses a heuristic transition from PSO to gradient-based search.

## ====================================================================
2)

de Rosa, Roder, Papa, dos Santos (2022), Improving Pre-Trained Weights Through Meta-Heuristics Fine-Tuning.
	=> PSO fine-tuning not actuall training

## ====================================================================
3)
Yosinski et al., 2014 How transferable are features in deep neural networks ?
	=> Early layers learn generic features transferable across tasks. This justifies using GD for the CNN backbone.

Quotes:
"In transfer learning, we first train a base network on a base dataset and
task, and then we repurpose the learned features, or transfer them, to a second target network to
be trained on a target dataset and task."
"We showed how transferability is
negatively affected by two distinct issues: optimization difficulties related to splitting networks in
the middle of fragilely co-adapted layers and the specialization of higher layer features to the original
task at the expense of performance on the target task."
"One can choose to backpropagate the errors from
the new task into the base (copied) features to fine-tune them to the new task, or the transferred
feature layers can be left frozen, meaning that they do not change during training on the new task."

"In order to compare transfer performance between tasks A and B such that A and B are as semanti-
cally dissimilar as possible, we sought to find two disjoint subsets of the 1000 classes in ImageNet
that were as unrelated as possible. "
	=> same dataset, split into disjoint subsets, train → reuse → adapt
"dataset A containing only man-made entities and B containing natural entities"
"“the first n layers are copied from a network trained on one dataset”"

## ====================================================================

4)
Parisi et al., 2019, Continual Lifelong Learning with Neural Networks: A Review
	=> This supports sequential learning from new datasets.

Quotes:
"The ability to continually learn over time by accommodating new knowledge while retaining previously learned experiences is referred to as continual or lifelong learning"
"Lifelong learning remains a long-standing challenge for machine learning and neural network models since the continual acquisition of incrementally available information from non-stationary data distributions generally leads to catastrophic forgetting."
"A lifelong learning system is defined as an adaptive algorithm capable of learning from a continuous stream of information, with such information becoming progressively available over time"
"Lifelong learning represents a long-standing challenge for machine learning and neural network sys-
tems (Hassabis et al. 2017, French 1999). This is due to the tendency of learning models to catas-
trophically forget existing knowledge when learning from novel observations"

A system that:
	learns multiple tasks sequentially
	from a continuous stream of data without forgetting previous tasks
	=> fix catastrophic forgetting problem	

## ====================================================================
5)
Communication-Efficient Learning of Deep Networks from Decentralized Data, H. Brendan McMahan

Quotes:
“We investigate a learning technique that allows users to collectively reap the benefits of shared models trained from this rich data, without the need to centrally store it.”
	=> means that learning happens without central storage of raw data.
	=> "“Each client has a local training dataset which is never uploaded to the server.”"
	=> "“Instead, each client computes an update to the current global model maintained by the server, and only this update is communicated.”"

"which are coordinated by a central server."
	=> “At the beginning of each round, a random fraction C of clients is selected, and the server sends the current global algorithm state to each of these clients.”

“The training data on a given client is typically based on the usage of the mobile device by a particular user, and hence any particular user’s local dataset will not be representative of the population distribution.”

“In federated optimization communication costs dominate”

“each client locally takes one step of gradient descent on the current model using its local data, and the server then takes a weighted average of the resulting models.”

"FederatedAveraging algorithm, which combines local stochastic gradient descent (SGD) on each client with a server that performs model averaging."
	=> This is the definition of FedAvg. Its called really The FederatedAveraging Algorithm

"Communication costs are the principal constraint, and we show a reduction in required communication rounds by 10–100× as compared to synchronized stochastic gradient descent."
	=> We do this instead of sync because communication effciency
	=> "can reduce the rounds of communication needed to train a deep network on decentralized data by orders of magnitude"

"on-IID data distributions that are a defining characteristic of this setting"

## ====================================================================
6)
Communication-Efficient Distributed Deep Learning via Federated Dynamic Averaging

"“Most DDL methods are iterative, where, in each iteration, some amount of local training is followed by synchronization of the local models with the global one.”"
	=> pure DL does syncing, not FedAvg

"The predominant method, based on the bulk synchronous parallel (BSP) approach [56], is to average the local model updates and then apply the average update to each local model [69]
"
“The communication bottleneck arises from the frequent exchange (synchronization) of model parameters, often in the range of billions, across distributed workers.”

“The most direct method to alleviate the communication burden is to reduce the frequency of communication rounds. Local-SGD is the prime example of this approach.”

“DDL algorithms typically waste valuable bandwidth… by relying on overly simplistic, periodic, and rigid synchronization schedules.”

“The costly synchronization step is triggered only if the local models… have significantly diverged.”

"The most direct method to alleviate the communication burden
is to reduce the frequency of communication rounds. "

" Our FDA algorithm dynamically triggers
synchronization based on the value of model variance across
worker-nodes. In a nutshell, the costly synchronization step is
only triggered if the local models have diverged significantly,
which implies that the global model may no longer be accurate."
	=> Synchronization still takes place but only when necessary.

## ====================================================================
7) Capel, capel
Parallel PSO for Efficient Neural Network Training Using GPGPU and Apache Spark in Edge Computing Sets

Quotes:
"This paper presents a novel approach to accelerating DLNN training using the particle swarm optimisation
(PSO) algorithm, which exploits the GPGPU architecture and the Apache Spark analytics engine
for large-scale data processing tasks."

"A new library in Scala has been developed with the objective of facilitating distributed training of neural networks using Spark."

“This allows the particle fitness computation and particle position update to be distributed across the different execution nodes of a standalone Spark cluster.”

“Two variants of Parallel PSO were proposed. DSPSO is more efficient for medium-sized datasets, while DAPSO is faster and more scalable for large datasets.”
	=> DSPSO = Distributed Synchronous Particle Swarm Optimization
	=> DAPSO = Distributed Asynchronous Particle Swarm Optimization
	"Synchronous PSO updates the swarm only after all particles have been evaluated, whereas asynchronous PSO updates particles as soon as evaluations become available."
	
"Despite the fact that these CUDA-based GPU PSO
acceleration proposals offer extremely low latency for computation and memory access, as
well as high throughput for suitable workloads, they suffer from poor scalability, which is
limited by the resources of a single GPU or a small number of GPUs"

The main point of this PSO implementation os to be scalable with increased data:
	"Accordingly, this algorithm will demonstrate superior performance in the handling and
	processing of massive datasets on distributed systems, exhibiting both scalability and fault tolerance."

	"In contrast, distributed implementations of PSO with Apache Spark are horizontally scalable, allowing
	for the addition of nodes to the cluster and thus making them suitable for processing very large datasets."

Uses Spark cluster scheduling rather than message-based communication. This means that Spark is inherently meant to be executed in a Cluster:

“Spark distributes the computational parallelisation process across the available cluster executors.”
"Secondly, the partitioning feature enables Spark to distribute the
computational parallelisation process across the available cluster executors."
"While the programs developed with Spark can be executed on diverse distributed platforms, this work exclusively
presents results from execution on a departmental GPU cluster. As future work, we intend to adapt the presented algorithms for deployment on alternative platforms, such as Kubernetes or Databricks."

“Particle fitness computation and particle position update are distributed across the execution nodes of the Spark cluster.”
	=> Uses actuall PSO training

## ====================================================================
8)
FedPSO: Federated Learning Using Particle Swarm Optimization to Reduce Communication Costs

Quotes:	

Communication Improvemnt with FedPSO
"Thus, we propose a FedPSO, a global model update algorithm with improved
network communication performance, by changing the form of the data that clients transmit to
servers.
	“We increase its robustness in unstable network environments by transmitting score values rather than large weights.”

	“The proposed model, FedPSO, receives the model weights only for the client that provided the best score so that the model weights do not need to be transmitted from all clients.”

"In this study, we propose the algorithm using particle swarm optimization algorithm instead of FedAvg,
which updates the global model by collecting weights of learned models that were mainly used in
federated learning."

“Applying FedPSO significantly reduced the amount of data used in network communication.”
	=> because of sending dcoes and ni updates

“FedPSO improved the accuracy of the global model by an average of 9.47%.”

“Both FedPSO and FedAvg used SGD methods for client training”
	=> This shows the clients are still trained with SGD, so PSO is not replacing the whole training process there.
	
PSO is used instead of FedAvg => PSO style server update rule:
	“Based on the weight update equation (Equation (2)), we present the conceptual algorithm of FedPSO.”
	“FedPSO identifies the best model through pbest and gbest variables and updates using the value of V"
	"Unlike conventional algorithms, Function ServerExecutes receives only pbest values, without receiving w from the client on Line 5."

## ====================================================================
9)
PSO-PS:Parameter Synchronization with Particle Swarm Optimization for Distributed Training of Deep Neural Networks

Quotes:
"could be iteratively updated during the model training using mini-batch stochastic gradient descent (SGD) optimizers and the back propagation algorithm"
	“To decrease the training time and utilize the gradients and the SGD optimizer … step is applied to make the optimizer switch between PSO and SGD.”
“a different subset of data is fed into different workers, and each worker performs its forward pass and backward pass individually”
“At each synchronization stage, the weights are updated by PSO from the sub weights gathered from all workers, instead of averaging the weights or the gradients.”

“the new parameters are calculated based on the particle updating mechanism, which is a completely different strategy from the existing methods that aggregate parameters or gradients at each synchronization”
“At each synchronization stage, the weights are updated by PSO from the sub weights gathered from all workers”
	sub-weights = locally trained weights from each worker.

## ====================================================================
10) Hammed Hamed
PARTICLE SWARM OPTIMIZATION FOR NEURAL NETWORK LEARNING ENHANCEMENT

Quotes:

"In this study, PSO is applied to feedforward neural network based on Al-kazemi
and Mohan [8], where the position of each particle in swarm represents a set of
weight for the current epoch or iteration."

<!-- ===================================================================== -->
=> PSO better than GA:
"Lee et al. [7] have used PSO and GA for excess return evaluation in
stock market. Based on their experiment, it is proven that PSO algorithm is better
compared to GA."

"Weights in BP are replaced with weights from GA, while η and α (learning and momentum rate) value in BP are replaced with value from GA process."
	=> BPNN

“PSO … can be effectively applied in neural network with faster convergence rate and promising classification accuracy compared to GANN.”

“The results show that PSONN give promising results in terms of convergence rate and classification accuracy compared to GANN.”
	=> GANN using backprop + SGD:
		- GA is used to optimize parameters / weights
		- Backpropagation (BP) is still part of the learning pipeline
	=> PSONN doesnt use Backprop

<!-- ===================================================================== -->

“There is no backpropagation concept in PSONN where the feedforward NN produced the learning error (particle fitness) based on set of weight and bias (PSO positions).”

“PSO is applied to feedforward neural network … where the position of each particle in swarm represents a set of weight for the current epoch or iteration.”

“The particle moves within the weight space attempting to minimize learning error. Changing the position means updating the weight of the network in order to reduce the error of the current epoch.”

“Train NN using new particle position” => means just Evaluate the NN again with new weights like extract errors / probabilities, not train it with SGD.

## ====================================================================
11)
Improving PreTrained Weights through Meta - Heuristics Fine- Tuning Train the model with GD and fine tunes it using PSO.

Quotes:

“In this work, we propose to use meta-heuristic techniques to fine-tune pre-trained weights, exploring additional regions of the search space, and improving their effectiveness.”

"The proposed approach aims to pre-train an architecture through its standard pipeline, e.g., stochastic gradient optimization across a training set, followed by a fine-tuning using meta-heuristic optimization across a validation set (post-trained)."\

In Table II: “PSO — w = 0.7 | c1 = 1.7 | c2 = 1.7”

## ====================================================================
12) Zhang, zhang
Particle swarm optimisation for evolving artificial neural network

Quotes:
“In PSONN system, the learning algorithm is the PSO algorithm.”
“Here the encoding scheme is that each individual is to parameterise a whole group of g nodes in ANNs, this means that every component of each individual represents a connection weight.”
“Generate an initial population of M networks.”
"GAS was used
to evolve ANNs, but the evolution of ANNs architectures
often suffers from the permutation problem [5]."
 => this may be relevant in PSO as well Two different weight vectors, can represent the same function
 => this creates redundancy in solution space:
	2 hidden neurons:Neuron A, Neuron B
	Now swap them: Neuron B, Neuron A
	The network output does not change at all

This one doesnt do full PSO training either (only NAS stuff):
"Using the PT algorithm to train each network’s nodes
on the training set (this process can evaluate the quality
of a given network architecture)."
	- PT = a partial training algorithm
	- a local training method applied to the network weights
	- Strong indirect evidence they use gradient-based training, they basically use Backprop
		PSO → explore architectures / candidate weights
		PT → train/refine the network

## ====================================================================
13) Kashyap
A Study of Hybrid and Evolutionary Metaheuristics for Single Hidden Layer Feedforward Neural Network Architecture

Quotes:
“This study presents a population-based metaheuristic framework for training ANNs … framing the training objective as a high-dimensional nonlinear optimization problem.”
“the trainable parameters of the ANN—consisting of weights and biases—are components of a high-dimensional vector.”
“This study employs a single hidden layer feedforward neural network architecture.”

“we present an innovative hybrid PSO–SGD approach that utilizes PSO’s global search capabilities and SGD’s local refining.”

“SGD modifies the weights using a first-order approximation of the loss landscape by descending down the negative gradient of the error function.” The protocol can be described as such:
	Neural Network (single hidden layer)
			↓
	Initialize weights
			↓
	Population of candidate weight vectors
			↓
	PSO / GA search
			↓
	Hybrid PSO-SGD update, which actually does:
		new_weights =
			PSO_velocity_step
		+ PSO_cognitive_term
		+ PSO_social_term
		- SGD_gradient_step

## ====================================================================
14) Wang, wang
Particle Swarm Optimisation for Evolving Deep Neural Networks for Image Classification by Evolving and Stacking Transferable Blocks

Quotes:
“In this paper, an efficient particle swarm optimisation method named EPSOCNN is proposed to evolve CNN architectures inspired by the idea of transfer learning.”
"The proposed method will be evaluated on the CIFAR-10 dataset ... In addition, the evolved block will be transferred and evaluated on two other datasets — CIFAR-100 and the Street View House Numbers (SVHN) dataset."
"In order to mitigate the bias introduced by only using a small subset, Adam optimisation [11] is used to train the CNNs instead of SGD optimisation [12]."
	=> Backprop

“the search space is minimised by integrating the existing expertise of hand-crafted CNNs”
	The architecture is not manually designed
	Even if only part of the architecture is searched
	So this is a restricted / reduced NAS approach

“In the proposed method, DenseNet [4] is used as the prior expertise to minimise the search space by encoding only the hyper-parameters of one dense block...”
	=> “Instead of evolving the whole network architecture, the PSO is only utilised to evolve the optimal Dense Block on the small subset.”
	=> NAS == Each particle is a proposal for how the CNN should be built.
		=> it encodes only a small architectural description of one dense block.
	
"PSO is used to design the architecture (NAS), while backpropagation is used to train the network during evaluation."
	=> "Furthermore, an adaptive training algorithm referred as Adam optimisation [11] is adopted to train the CNNs during the fitness evaluation"
	=> PSO needs to evaluate architecture ... to do so, the architecture needs to be first trained, then PSO can evaluate it and then back to PSO search space 

“Thirdly, an automatic and progressive process of stacking the learned block is proposed to increase the capacity of the final neural network.”
"the proposed method stacks the learned block multiple times to obtain a CNN with more capacity, which will be depicted in Section III-E."
	- stacking: repeating the same neural network block multiple times to build a deeper network 
		- [Block] → [Block] → [Block] → output
	- Core idea:
		Search for one good CNN building block
		Then stack it multiple times
		That becomes the final architecture
“the learned block from a smaller dataset is transferable to a larger dataset”
	It suggests transfer learning (weights)
	But actually it’s architecture reuse	=> transferable == reusable

Step 1 — PSO proposes a candidate
Particle = (layers, growth rate)
Defines a CNN block
Step 2 — Train that candidate (THIS is backprop)

They:

build the CNN block
train it using Adam (gradient descent)

Quote:

“Apply Adam optimisation to train block”

Step 3 — Evaluate performance
Measure accuracy
This becomes the fitness
Step 4 — PSO updates particles
Based on fitness
Moves toward better architectures
Step 5 — Repeat


## ====================================================================
15)
Top-k Multiclass SVM

I need this for topk (this one is per sample)

## =====================================================================
16)
https://lightning.ai/docs/torchmetrics/stable/classification/hinge_loss.html?utm_source=chatgpt.com

For multi class hinge loss

## =====================================================================
17)
DERIVATIVE-FREE OPTIMIZATION FOR CUSTOM LOSSFUNCTIONS

"Derivative-free optimization (DFO) has emerged as a powerful technique for solving optimization problems where the gradi-
ent of the objective function is either unavailable, expensive to compute, or non-smooth."
"“in many real-world applications ... Custom loss functions may be requi-red when dealing with domain-specific objectives, non-traditional evaluationmetrics, or real-world constraints that are difficult to express in a differentiable form."
"Examples include loss functions that depend on ranking metrics (suchas mean average precision in information retrieval), loss functions involvingdiscrete variables (such as edit distance in text generation), or loss functionsthat integrate external black-box evaluations (such as real-world reinforcementlearning rewards or adversarial robustness measures)"
"However, in many real-world applications, the use of standard loss functi-ons is insufficient or even inappropriate"
“the generator trained with the p-statistic loss produced diverse and realistic images” => Training GAN with it
“derivative-free optimization provides an essential toolset for handling complex, custom loss functions”
	BUT:
		It’s slower
		More computationally expensive
“loss functions may be… non-differentiable… or defined through external processes such as simulations”

## ====================================================================
18)
The Fully Imformed Particle Swarm: Simpler, Maybe Better

"Random weighting of the two terms
keeps the particle searching between and beyond a re-
gion defined by the two points"
“Thus, neighborhood size determines how diverse your influences will be and in an optimization algorithm diverse influences might mean that search is diluted rather than enhanced.”
“As expected, increasing the size of the neighborhood seems to deteriorate the performance of the swarm.”
“The very worse FIPS conditions in the study were the UAll and All topologies, where the particle is truly fully informed, gathering information from every single member of the population.”

No Free Lunch:
“NFL asserts that no algorithm can be better than any other, over all possible functions.”
“It does not seem interesting to us to demonstrate that an algorithm is good on some functions and not on others.”
	=> An algorithm should be good at many benchmarks / functions
	=> “What we hope for is a problem-solver that can work well with a wide range of problems.”

“NFL does not say that the search for a general problem-solver is futile; it does say that the search for a general function optimizer is futile.”

## ====================================================================
19)
On the Scalability of Particle Swarm Optimisation
	
“However, like any optimization algorithm it seems to have difficulties handling optimization problems of high dimension.”
“Here we first show that dimensionality is really a problem for the classical particle swarm algorithms.”
“We then show that increasing the swarm size can be necessary to handle problem of high dimensions but is not enough.”
“To solve problems of increasing size, it is necessary to increase the swarm size and to run for more iterations, but this is not always sufficient to solve the problem in 10,000,000 evaluations.”

## ====================================================================
20)
No Free Lunch Theorems for Optimization

“for any algorithm, any elevated performance over one class of problems is offset by performance over another class.”
“it has become important to understand the relationship between how well an algorithm performs and the optimization problem on which it is run.”
“how can we best match algorithms to problems”
“the average performance of any pair of algorithms across all possible problems is identical.”
“These same results also indicate the importance of incorporating problem-specific knowledge into the behavior of the algorithm.”

## =====================================================================
## Picture Gathering:
0)
https://medium.com/%40minervaaniket/apache-kafka-a-deep-dive-into-its-architecture-and-workflow-510709dff298
1)
https://www.instaclustr.com/education/apache-kafka/apache-kafka-architecture-a-complete-guide-2026/
2)
https://cloud.ibm.com/docs/EventStreams?topic=EventStreams-apache_kafka
3)
https://www.geeksforgeeks.org/apache-kafka/kafka-architecture/
4)
https://docs.confluent.io/platform/current/streams/architecture.html#processor-topology
5)
https://www.analyticsvidhya.com/blog/2022/08/exploring-partitions-and-consumer-groups-in-apache-kafka/
6)
https://developer.confluent.io/courses/kafka-streams/internals/
8)
https://www.jasss.org/25/2/8.html
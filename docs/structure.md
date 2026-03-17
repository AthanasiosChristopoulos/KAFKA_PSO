
## ================================================================================================

# Table of Contents

# 1 Introduction
	=> how is this overal field relevant in todays world => Broad context / global trend	
		=> Give 2–4 concrete domains where the problem appears
	=> Explain why the problem is difficult
	=> Briefly mention existing frameworks / technologies, 
		=> and why this doesnt solve the problem completely are
	=> Introduce your system / method

	=> what does this work contribute

	- Broad context / global trend

Context
	Big Data / ML / distributed systems trend
	Why current approaches struggle

Existing technologies
	Kafka, Spark, etc.

Research gap
	What is missing

This thesis
	What you propose

## 1.2 Motivation
### 1.2.1 Why Particle Swarm Optimization (PSO)
### 1.2.2 Why Federated / Distributed Processing Environment
### 1.2.3 Why Kafka / Kafka Streams as Architecture
## 1.3 Thesis Scope and Contributions
## 1.4 Motivation
## 1.5 Thesis Outline

# 2 Motivation

# 3 Related Work
	Papers που κάνουν παρόμοια δουλεια με αυτην (biger picture δουλεια - το distributed PSO)
	Papers που λύνουν μέρος του προβλήματος
	Papers που είναι βάση για της δουλειά
	
## 3.1 PSO Basics
## 3.2 PSO in Neural Learning
## 3.3 Hybrid PSO-SGD training
## 3.4 Distributed and Federated Neural Learning using PSO
## 3.5 Existing Literature Gaps
## 3.6 Existing Literature Gaps

# 4 Thesis Contribution

# 5 Theoretical Background
## 5.1 Kafka and Kafka Streams
### 5.1.1 Kafka Fundamentals
### 5.1.2 Kafka Retention Policies
### 5.1.3 Kafka Streams
## 5.2 Classification Tasks and Neural Networks
### 5.2.1 Classification
### 5.2.2 Neural Networks
### 5.2.3 MLPs
### 5.2.4 Loss Functions
### 5.2.5 CNNs
## 5.3 Transfer Learning
## 5.4 Federated and Distributed Learning
### 5.4.1 Distributed Learning (DL) Basics
### 5.4.2 Federated Learning Concepts
## 5.5 Particle Swarm Optimization (PSO)
### 5.5.1 Classical PSO
### 5.5.2 Neighborhood Best vs Global Best
### 5.5.3 Fully Informed PSO (FIPS)
### 5.5.4 PSO used for Neural Network Training

# 6 System Requirements
## 6.1 Non-Functional Requirements
### 6.1.1 Scalability
### 6.1.2 Extensibility
### 6.1.3 Portability
### 6.1.4 Reliability
## 6.2 Functional Requirements
### 6.2.1 Dataset Streaming and Processing
### 6.2.2 Distributed PSO Training
### 6.2.3 Forward Pass Capability and Fitness Calculation
### 6.2.4 Inference Capability

# 7 Implementation
## 7.1 Protocol
### 7.1.1 Baseline PSO Protocol
### 7.1.2 Distributed / Kafka-based PSO Protocol
## 7.2 System Architecture
### 7.2.1 Kafka Consumers and Producers
### 7.2.2 Kafka Topics
### 7.2.3 Kafka Message Types
### 7.2.4 General Data Flow
### 7.2.5 Two PSO communication modes
### 7.2.6 Implementation of neighborhoods in Kafka-based PSO
### 7.2.7 Coordinator monitoring, evaluation, and inference pipeline
### 7.2.8 Kafka Streams Tranformers
## 7.3 Stochastic Gradient Descent vs PSO for Neural Network Training
### 7.3.1 Effect of Dimensionality
## 7.4 PSO Techniques Used
### 7.4.1 Adaptive Inertia
### 7.4.2 Time-varying acceleration coefficients
### 7.4.3 Velocity clamping
### 7.4.4 Performance Weights on update
## 7.5 Communication Filters
### 7.5.1 Debounce Filter
### 7.5.2 Significant Loss Filter
### 7.5.3 Monitoring Filter
## 7.6 Loss Functions
### 7.6.1 Non-Differentiable Loss Functions
### 7.6.2 Hinge Loss
### 7.6.3 Top-k Mean Absolute Error
### 7.6.4 Sorted L_1 Penalized Estimator Regularization

# 8 Experimental Evaluation
## 8.1 Experimental Setup
### 8.1.1 Kafka Setup
### 8.1.2 Hardware and Software
## 8.2 Performance Considerations
### 8.2.1 System Bottlenecks
## 8.3 Parameter Experimentation
### 8.3.1 Fully Informed vs classical PSO
### 8.3.2 Model Dimensionality vs Training Accuracy
### 8.3.3 Topologies
### 8.3.4 Number of Workers
### 8.3.5 Filter Parameters
## 8.4 Exploration of different Datasets
### 8.4.1 Datasets and Models
### 8.4.2 Iris
### 8.4.3 WineQuality
### 8.4.4 Pendigits
### 8.4.5 MNIST5
### 8.4.6 MNIST
### 8.4.7 CIFAR5
### 8.4.8 Non Differentiable Loss Function Experimentation
### 8.4.9 Discussion

# 9 Future Work
## 9.1 PSO Algorithmic Improvements
## 9.2 System Improvements
## 9.3 More Experimentation & Testing
## 9.4 Memory Efficiency

	
## =====================================================================================================

	Thesis Structure:
		- Acknowledgements
		- Abstract
		- Table of Contents
		- Introduction (Why is this a current world problem, why this field of research has a problem in the modern world and what is my solution to this problem):
			- Need to study PSO:
				- Non Differentiable Function
			- Federated / Distrubuted Processing enviroment
			- Why i choose Kafka / Kafka Streams as the architecture 
			- What this work is and how this project contributes to it

		- Motivation
		- Related Work 
		- Thesis Contribution

		- Theoretical Background => is this generall in nature or what is necessary to understand before proceding ???:
			- Kafka / Kafka Streams (copy paste ???)
			- Classification Task Explained + Neural Networks:
				- FNNs / MPLs
				- CNNs	
			- Transfer Learning
			- Federated Learning (Distributed Learning ???)
			- PSO

		- Functional and Non Functional System Requirements
			- εξηγω τι χρειαζεται να κανει το προγραμμα μου πριν εξηγησω πως το κανει

		- Implementation:
			- Protocol (Differences between normal PSO protocol and distributed / Federated learning / Kafka PSO protocol)
				=> βαλε το PSO (original and yours) pseudo algorithm ως προτοκολλο
				=> many things, like the sharing and updating of gBest, can be easily done on the same device.
				=> that isnt the case with federated, communication is costly 
			- Architecture (Kafka Topology, Project Class Hierarchy)
				- Exact topology / Kafka Details explained 
				- Trandformes explained
			- Difference between GD and PSO
				- how much dimensionality affects PSO
			- Different Non - Differentiable Loss Functions. You need to demonstrate about 3 such functions and show that they are non differentiable and the performance while using them.
			- PSO techniques used:
				- choose between them using .env interface => "User Interface"
			- Performance of the program (how the forward pass needs to be the bottleneck and that it doesnt matter if you add extra work to the transformers)

		- Experimental Evaluation:
			- Datasets it run on with their corresponding models
			- Special Transfer learning stuff for CNN Datasets and Excuses
			- Measure Accuracy and Performance:
				- Run it locally 
				- Run it on a server
				- Using different .env stuff 

		- Future Work
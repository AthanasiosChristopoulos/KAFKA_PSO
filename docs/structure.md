
## ================================================================================================

# Table of Contents

## 1. Introduction
### 1.1 Problem Context and Relevance
### 1.2 Proposed Solution Overview
#### 1.2.1 Why Particle Swarm Optimization (PSO)
#### 1.2.2 Why Federated / Distributed Processing Environment
#### 1.2.3 Why Kafka / Kafka Streams as Architecture
### 1.3 Thesis Scope and Contributions
### 1.4 Motivation
### 1.5 Thesis Outline

## 2. Related Work
### 2.1 PSO in Neural Learning
### 2.2 Distributed / Federated Neural Learning
### 2.3 Kafka-based Training / Inference Pipelines
### 2.4 Existing Literature Gaps

## 3. Thesis Contribution
### 3.1 Assumptions and Limitations
### 3.2 Expected Experimental Outcomes

## 4. Theoretical Background
### 4.1 Kafka and Kafka Streams
#### 4.1.1 Kafka Fundamentals
#### 4.1.2 Kafka Retention Policies
#### 4.1.3 Kafka Streams Concepts

### 4.2 Classification Tasks and Neural Networks
#### 4.2.1 Classification
#### 4.2.2 Neural Networks
#### 4.2.3 Multi-Layer Perceptrons (MLPs)
#### 4.2.4 Loss Functions
#### 4.2.5 Convolutional Neural Networks (CNNs)

### 4.25 Transfer Learning

### 4.3 Federated and Distributed Learning
#### 4.3.1 Distributed Learning Basics
#### 4.3.2 Federated Learning Concepts

### 4.4 Particle Swarm Optimization (PSO)
#### 4.4.1 Classical PSO
#### 4.4.2 Neighborhood Best vs Global Best
#### 4.4.3 Fully Informed PSO (FIPS)
#### 4.4.4 PSO for Neural Network Training

## 5. System Requirements

## 6. Implementation
### 6.1 Protocol
#### 6.1.1 Baseline PSO Protocol
#### 6.1.2 Distributed / Kafka-based PSO Protocol

### 6.2 System Architecture
#### 6.2.1 Kafka Consumers and Producers
#### 6.2.2 Kafka Topics
#### 6.2.3 Kafka Message Types
#### 6.2.4 General Data Flow
#### 6.2.5 Two PSO Communication Modes
#### 6.2.6 Implementation of Neighborhoods in Kafka-based PSO
#### 6.2.7 Coordinator Monitoring, Evaluation, and Inference Pipeline

### 6.3 Gradient Descent vs PSO for Neural Network Training
#### 6.3.1 Effect of Dimensionality

### 6.4 Non-Differentiable Loss Functions
#### 6.4.1 Loss Function A
#### 6.4.2 Loss Function B
#### 6.4.3 Loss Function C

### 6.5 PSO Techniques Used
#### 6.5.1 Adaptive Inertia
#### 6.5.2 Time-varying Acceleration Coefficients
#### 6.5.3 Velocity Clamping
#### 6.5.4 Performance Weights on Update

### 6.6 Communication Filters
#### 6.6.1 Debounce Filter
#### 6.6.2 Significant Loss Filter
#### 6.6.3 Monitoring Filter

### 6.7 Miscellaneous

## 7. Experimental Evaluation
### 7.1 Performance Considerations
#### 7.1.1 Forward Pass as Bottleneck
#### 7.1.2 System Overhead

### 7.2 Experimental Setup
#### 7.2.1 Hardware and Software
#### 7.2.2 Test Dataset and Model
#### 7.2.3 Fully Informed vs Classical PSO
#### 7.2.4 Increasing Number of Workers
#### 7.2.5 Using and Adjusting Filters
#### 7.2.6 Final Recommended System Requirements

### 7.3 Execution
#### 7.3.1 Datasets and Models
#### 7.3.2 Final Results
#### 7.3.3 Discussion

## 8. Future Work
### 8.1 PSO Algorithmic Improvements
### 8.2 System Improvements
### 8.3 Further Experimental Testing
### 8.4 Saving Memory PSO


## Appendix A: Additional Material
### A.1 Fully Informed Velocity Update Mathematical Proof
### A.2 Extra Tables and Figures
### A.3 Configuration Examples


	
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
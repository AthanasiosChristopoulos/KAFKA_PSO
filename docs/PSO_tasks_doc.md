
## =======================================================================================
## Important Pending Tasks:
	
	1) Find better Filters:
	
		Geometric Monitoring:
			- Actuall GM:
				- Synchronization necessary
			- παραλλαγης στο gm:
				- ενας worker i βλεπει οτι κανει cross το border / τo A / φευγει απο το κατωφλι
				- στελνει μονο αυτος στον Coordinator, ο οποιος κανει update το global average με update μονο του vector του i (οι αλλοι μενουν σταθεροι)
				- το καινουργιο global average στελνεται ξανα σε ολους για sync 
				- γλιτωνουμε απο την upstream επικοινωνια με coordinator, οχι ομως με την downstream
			- χρηση παραλλαγης στο gm, αλλα χωρις syncs, δηλαδη να δουλευει asynchrously: 
				- μονο οταν ενας worker υπερβαινει / κοβει το Α, τοτε μονο αυτος στελνει (no coordinator sync)
				- indicating that this update will cause a significant difference in the FedAvg / globalAverage
		
			- για GM, θα μπορουσα να κανω το heuristic να δουλευει (με FFT conversion), απλως full GM ως προτοκολλο εχει νοημα μονο για synchroniztion, 
				που δεν μπορει να το κανει το Kafka Streams:	
			- το προτοκολλο ειναι fully asynchronous. Δηλαδη δεν μπορω να σταματησω το training και να αρχισω να περιμενω για καινουργιο global μοντελλο
				- οσο ενας worker περιμενει νεα weights, αναγκαστηκα θα κανει consume νεα δεδομενα με το παλιο μοντελλο (no stop and weight).
			- το sync στο PSO ενδεχεται να ειναι αντιφατικο, καθως στο PSO περιμενουμε το "sync" να ερθει ως αποτελεσμα του swarm να κανει converge, οχι να γινει enforced.
				- Για να γινει αυτο τα particles κανουν maintain το δικο τους position, δεν κανουν sync
		
		Limit communication στο local_weights_topic pipeline:
			- Federated Learning component of your project, averaging (FedAvg) and καθε ποσο κανω track το position
				- μεχρι τωρα, καθε 60 updates. Θα πρεπει να αλλαξει αυτο, να κρινεται απο το ποσο εχει κανει diverge
				- Measure of distance με προηγουμενη κατασταση (αμα εχει αλλαξει significantly ή κατασταση)

		Prediction models σημαινει προβλεψη pbest / gbest, ωστε να ξερω αμα θα επικοινωνησω ή οχι

	# ========================================================================================

	66) Experimentation - We want following diagrams:

		- 0) x: classical / fully_informed	| y: accuracy	=> conclusion: fully informed better
		- 1) x: Dimensionality 				| y: accuracy	=> conclusion: high dimensionality bad
		- 2) x: Different Topologies 		| y: accuracy	=> Trade off between neighborhoods

		- 3) x: N_WORKERS  					| y: accuracy / time / number of bytes
		- 4) x: FILTER_STRENGTH 			| y: accuracy / time / number of bytes
			- Δεν πειραζει ο χρονος να μην πεφτει εχουμε bottlenecks τα forward passes και το Disk / Broker I/O
			If not:
			=> x: FILTER_ENABLED 			| y: accuracy / time / number of bytes
			=> x: threshold T				| y: accuracy / number of bytes
		- 5) x: N_WORKERS  					| y: accuracy / time / number of bytes (with Filter)

			=> Repeat 3, 4, 5 for all 4 Datasets (also for the one from mnist to fmnist) (ειναι 3 απλα datasets και fmnist <--> mnist)

		- Ευρεση SWEET SPOTS => υπαρχει ενα trade-off σε καποια parameters δεν ειναι flattly good, ποιο ειναι το καλυτερo number of workers
			- N_WORKERS ανεβαζει το accuracy, αλλα ανεβαζει επισης τα Bytes sent	

		- End) x: Iterations 				| y: Accuracy	(with recommended SWEET SPOTS, filter and N_WORKERS ?)
			=> For all Datasets (and cases of interest)

		- Final Part 3) Different Non Differentiable Loss Functions
			- accuracy - monitoring roinds diagram

	# ========================================================================================

	57) PSO Tranfer Learning: This is transfer learning between GD and PSO, where last layer never touches GD, because we want non differentiable Transfer Function. Try harder on the pretrained model, in these levels:

		- A) Remove more Layers
		- B) Unfreeze / Fine-Tune base network (train the highly specific / class specific / last conv layers (paper))
		- C) Semi train pretrained model, rather than fully train (smaller amount of epochs)
			=> Goal is to achieve greater performance comparing the pretrained version of the model to the newly PSO-trained version of the model 
		- D) Use models that are not trained in the same dataset and are repurposed by tranfer learning
		- E) Use ImageNet models

		=> go from 1 Output Layer to 2 Dense Layers. Θελει περισσοτερα Layers (2-3)
		=> το (C) να χρησιμοποιεις pretrained μοντελλο πανω στο ιδιο dataset ειναι ξεχωριστο scenario οχι αναγκαστηκα προβλημα

		Solution_1: 
			- Find a model in GD (trained in tensorflow) that doesnt perform well (below 70% on MNIST) and has no dense layer (it relies solely on Convolutional Layer)
			- Go to DL4J / PSO and add those Layers achieving a higher accuracy
			- The idea is that GD will create a great CNN feature extractor and PSO will be able to use it and training its own classification layer on top of it
				=> Goal is for PSO to be of significant improvement (accuracy GAP to be significant) and if possible to have PSO increase accuracy beyond 90%
				=> Problems: PSO suffers under high dimensionality, the more dense layers with higher parameters the worse the result. Also PSO basically cant train CNNs, so conv layers in the head are out of the question
				=> GD backbone (good features, weak classifier), PSO head (small dense layer)
				=> Conv features that are good, but final decision boundary is suboptimal
				=> no ImageNet models
				
		Solution_2:
			- Find a Dataset that is similar, but different to the target dataset
			- Train the model on that dataset
			- Transfer it to the other dataset, replacing its classification layers and using the other data + PSO to train it
			- no ImageNet models

			due to the nature of PSO this is not possible  to use ImageNet type models 
			 - Hard dataset → GD pretraining
			 - Easier dataset → PSO adaptation

	# ========================================================================================

	64) More Transfer Learning Experimentation:

		"Self Transfer Learning":
		- Not transfer learning (by definition means transfering from task 1 to task 2)
		- του βαζω ενα ποσοστο των δεδομενων στο pretraining => historic data και μετα τα υπολοιπα που δεν εχει δει => previously unseen
		- Motivation for self transfer learning:
			- Αλλαξε η συναρτηση
			- Αλλαξαν τα δεδομενα
			- Εφαρμογη ενος προεκπεδευμενου μοντελλου (πανω σε ιστορικα δεδομενα - Historic Data) που μετα το β
			- Ειδικο scenario transfer learning

		- Αρα δοκιμαζεις:
			- CIFAR5 <--> CIFAR10 και CIFAR10 <--> CIFAR10 
			- New Image Dataset of difficulty between MNIST and Cifar10

		Transfer learning with temporal domain shift or continual learning.
			- Sequential Transfer Learning / Continual Learning
			- Pretraining → Fine-tuning on the same dataset distribution
			- Self-training / Self-transfer (rare term)

		Train on dataset 𝐷ℎ𝑖𝑠𝑡𝑜𝑟𝑖𝑐, then adapt to 𝐷𝑛𝑒𝑤, this is known as:
			=> Continual Learning (CL)
			=> Lifelong Learning

		Continual learning studies how models can learn from a sequence of data distributions without forgetting previous knowledge.

		Domain Adaptation
			Source domain: historic dataset
			Target domain: new unseen dataset
		
		Sequential training with different optimization methods on disjoint data splits:
			Dataset D
			├── Historic subset D₁ (50%)
			│      train with GD
			│
			└── Unseen subset D₂ (50%)
					continue training with PSO

		Warm-start training:
			You start optimization from pretrained weights instead of random initialization.

		You must not leak the same samples between phases. 

	## =======================================================================================

	65) Find better Non differentiable Functions, with requirements:
		- 1) Continious, but non differentiable
		- 2) Πρεπει να εχουν χρησιμοποιηθει σε καποιο γνωστο / οχι οτι να ναι venue (χώρος δημοσίευσης)
		- 3) Πρεπει οταν τα χρησιμοποιω να μην πεφτει πολυ το accuracy σε συγκριση με cross entropy

	## =======================================================================================

	69) Prediction Models: Best one is Static
		
	## =======================================================================================
	## New Tasks

	75) You obviously didnt read all these papers ...
		=> clean them up

	# ========================================================================================

	++) If more work is necessary:

		Low Priority ==============================================
		
		- Better / more complex filters
		- Better solution than cifar or mnist transfer learning
		- read other peoples diplomatiki and other peoples papers to tell if yours is good enough ...

		Lower Priority ============================================

		- Better Non Differentiable functions
		- Use of PSO techniques for better accuracy
		- Better starter datasets
		
	# ========================================================================================

	77) What tasks, that are not classifications, are non differentiable functions useful for, meaning they beat cross entropy
		- Generally prove that there is no dominant loss function for every tasks (no free lunch)

## =======================================================================================

	Backlog Tasks:
	
	6) Related Work + Research:
		- Other attempts implementing PSO
		- how does yours differ ?
			- Implementation / Enviroment / Technologies (Docker, Java, Kafka Streams)
			- What scientific differences did you add / remove / what parameters did you use ?
	
	19) velocity initialiazation (magnitude)

	13) Figure out Kafka Streams aggregation HOF thing (straight from Kafka Streams DSL)

	11) Add buffer as a statestore (on worker who is accumulating samples)

	17) X_G state store - the global current position

	21) Use Vector class not float[] => important for calculations

	29) Have both training and test accuracy => really hard on the coordinator because it relies on loading the entire test dataset.
	
	41) Συγκριση με gradient descent (σε ολα τα datasets) => too depressing
		Συγκριση η τελικη θα γινει με κατανεμημενο περιβαλλον στο training του gradient descent:
		- οχι centralized, αλλα gradient descent. 
		- Δηλαδη θελουμε να συγκρινουμε PSO vs gradient descent σε distributed περιβαλλον
		- Δες parameter server (independent learning με merging των μοντελλων afterwards) => δεδομενα ειναι λιγοτερα για καθε worker
		- Basically πρεπει να κανεις το federated learning στην python με gradient descent

	52) End-to-end propagation delay


## ================================================================================================

 - Generall Plan for now (5-43:36):
	- 5 Datasets:
		- iris (winequality - has 6500 samples)
		- susy
		- pendigits
		- MNIST 	(and / or MNIST5)
		- CIFAR4 	(and / or CIFAR)
	- On them try out:
		- baseline (no filtering)
		- with filter
	- 3 Non-Differential Functions


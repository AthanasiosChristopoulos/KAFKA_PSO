
## =========================================================================
## Generall Aspects of this Thesis that need Improving / Expanding upon (and how they are combined with each other):

 - PSO
 - Neural Networks - Models Used for what datasets
 - Kafka / Kafka Streams
 - Filtering / Efficient Communication Protocol
 - Federated Learning

## =======================================================================================
## Important Pending Tasks:
	
	1) Limit communication / Communication Efficiency:
	
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
		
	40) You need to measure exactly how much communication costs time wise (like you did for Kafka record processing / forward passes)
		- communication here isnt data record reading, but weight messages exchanged between coordinator and worker

	2) Experimentation:
		- Different models
		- Different Non - Differentiable Loss Functions
		- Different Datasets, Different sizes: INDEPENDENT_WORKER_DATA_PROCESSING true or false
		- Measure performance and latencies
		- Explain the Experimentation Setup:
			- Hardware (Laptop / Server (Cloud) / Raspberry Pis)
				- Actually mention / analyse the specs
				- Run on both highend and lowend devices 
			- Use and test as many different:
				- model / method parameters as possible
				- measurement methods (error definitions) 
			
		- Measure / Diagrams of (For these kinds of experiments set a paradigm where the samish correct solution is found => need data and convergence criteria):

			- x: training samples / epochs / updatesX 	| y: loss / accuracy / F1 score	
				- based on data you need to adjust parameters every time (find an automatic function for that)
			- x: N_WORKERS 								| y: accuracy / time
			- x: N_WORKERS 								| y: number of messages / number of bytes

			- Number of workers => increases parallelization / speed (throughput) - how much data the pipeline can process, bytes (records) per second
			- how much time on average does it take you to process one record 
				- this is supposed to be a streaming application: input_mbps < process_mbps
			- Limit communication:
				- Run without => identify latency (and number of messages)
				- Run with	  => makes training more efficient / faster, identify reduction in number of messages.

			- This is affected by Model Size (Number of Weights - NN?K) / Number of workers / BATCH_SIZE / FULLY_INFORMED (PROTOCOL used) / Loss Function:
				- How much time until reached DESIRED_ACCURACY (Training Time) ?
				- How much data until reached DESIRED_ACCURACY - How many epochs ?
				- Αμα το αφησεις να παει οσο παει, τοτε πιο ειναι το ελαχιστο loss / μεγιστο Accuracy που μπορει να φτασει ?
			- Performance and Accuracy Comparison with Gradient Descent

			- Με απλα Datasets:
				- Sweet spot of N_WORKERS (να σταματησει να αυξανει το accuracy significantly, τοτε δεν εχει νοημα η αυξηση του N_WORKERS, καθως αυξανουμε την επικοινωνια):
					- Fixed Threshold
					- x: N_WORKERS / y: accuracy vs communication(Number of messages)  
				
				- Threshold Sensitivity (Also Sweet Spot of threshold): 
					- Fixed N_WORKERS
					- Define T as: T = MONITORING_THRESHOLD_MAX - MONITORING_THRESHOLD_MIN;
					- x: threshold T (increasing) | y: accuracy (hopefully decreasing)
					- x: threshold T (increasing) | y: communication (hopefully decreasing)
					- Decreasing => strict to loose: This is about how aggressive your communication filtering is
						- Strict: Hard to pass the filter, LESS communication
						- Loose: Easy to pass the filter, MORE communication

			- x: Dimensionality / y: Accuracy

	43) PSO Tranfer Learning:
		- Ευρεση καταλληλου base model for MNIST and CIFAR + trainable End Layers
		- Θα πρεπει να βρεις additional Layers + Non Differentiable Loss Functions, ωστε:
			- το base model να μην δουλευει καλα
			- να κανεις train το frozen base model  το trainable End Layers, ωστε να δουλευει καλυτερα απο το σκετο base model
			- συνηθως δεν πας να κανεις train from scratch

	57) This isnt fully GD / PSO hybrid. This is transfer learning between GD and PSO, where last layer never touches GD, because we want non differentiable Transfer Function. Try harder on the pretrained model, in these levels:
		- A) Remove more Layers
		- B) Unfreeze / Fine-Tune base network
		- C) Semi train pretrained model, rather than fully train (smaller amount of epochs)
			=> Goal is to achieve greater performance comparing the pretrained version of the model to the newly PSO-trained version of the model 
		- D) Use models that are not trained in the same dataset and are repurposed by tranfer learning
		=> go from 1 Output Layer to 2 Dense Layers. Θελει περισσοτερα Layers (2-3)
		=> το (C) να χρησιμοποιεις pretrained μοντελλο πανω στο ιδιο dataset ειναι ξεχωριστο scenario οχι αναγκαστηκα προβλημα

		Solution: 
			- Find a model in GD (trained in tensorflow) that doesnt perform well (below 70% on MNIST) and has no dense layer (it relies solely on Convolutional Layer)
			- Go to DL4J / PSO and add those Layers achieving a higher accuracy
			- The idea is that GD will create a great CNN feature extractor and PSO will be able to use it and training its own classification layer on top of it
				=> Goal is for PSO to be of significant improvement (accuracy GAP to be significant) and if possible to have PSO increase accuracy beyond 90%
				=> Problems: PSO suffers under high dimensionality, the more dense layers with higher parameters the worse the result. Also PSO basically cant train CNNs, so conv layers in the head are out of the question
				=> GD backbone (good features, weak classifier), PSO head (small dense layer)
				=> Conv features that are good, but final decision boundary is suboptimal
	58) κανε το 32X32 => 224Χ224 Conversion in RAM, οχι στο broker

## =======================================================================================

	Backlog Tasks:

	3) Learn more about Kafka Streams + Tensorflow + Neural Learning
		- Improve code on Apache Kafka + Tensorflow:
		- Use / Learn about Parallelization (Threading / Instances):
				- On Kafka Streams: Threading and more Kafka Streams instances
				- Kafka Broker: More brokers
	
	6) Related Work + Research:
		- Other attempts implementing PSO
		- how does yours differ ?
			- Implementation / Enviroment / Technologies (Docker, Java, Kafka Streams)
			- What scientific differences did you add / remove / what parameters did you use ?
	
	19) velocity initialiazation (magnitude)

	21) Implement neighborhood topologies more efficiently:
		=> Problem: With current protocols, if there are N Workers, there are N neighborhoods
		=> Change Kafka / Kafka Streams architecture with Routers to fascilitate neighborhoods

	13) Figure out Kafka Streams aggregation HOF thing (straight from Kafka Streams DSL)

	11) Add buffer as a statestore (on worker who is accumulating samples)

	17) X_G state store - the global current position

	21) Use Vector class not float[] => important for calculations

	29) Have both training and test accuracy => really hard on the coordinator because it relies on loading the entire test dataset.
	
	41) Συγκριση η τελικη θα γινει με κατανεμημενο περιβαλλον στο training του gradient descent:
		- οχι centralized, αλλα gradient descent. 
		- Δηλαδη θελουμε να συγκρινουμε PSO vs gradient descent σε distributed περιβαλλον
		- Δες parameter server (independent learning με merging των μοντελλων afterwards) => δεδομενα ειναι λιγοτερα για καθε worker
		- Basically πρεπει να κανεις το federated learning στην python με gradient descent

	47) Κανε share one drive με γραφικες (διαγραμματα και τετοια ...)

	51) Get ONXX - Pytorch - Cifar - 32 x 32 x 3 models

	52) End-to-end propagation delay
	
## =====================================================================================================

	Thesis Structure:

		- Motivation, Table of Contents
		- Related Work / Thesis Contribution
		- Introduction (Theoretical Background)
		
		- Implementation:
			- Architecture (Kafka Topology, Project Class Hierarchy)
			- Difference between GD and PSO
				- how much dimensionality affects PSO
			- Protocol (Differences between normal PSO protocol and distributed / Federated learning / Kafka PSO protocol)
				=> many things, like the sharing and updating of gBest, can be easily done on the same device.
				=> that isnt the case with federated, communication is costly 
			- PSO techniques used
			- Different Non - Differentiable Loss Functions. You need to demonstrate about 3 such functions and show that they are non differentiable and the performance while using them.
			- Performance of the program (how the forward pass needs to be the bottleneck and that it doesnt matter if you add extra work to the transformers)
			- Functional and Non Functional Requirements

		- Experimental Evaluation:
			- Run it locally 
			- Run it on a server
			
		- Future Work

## ================================================================================================

 - Generall Plan for now (5-43:36):
	- 5 Datasets:
		- iris (winequality - has 6500 samples)
		- susy
		- pendigits
		- MNIST 	(and / or MNIST4)
		- CIFAR4 	(and / or CIFAR)
	- On them try out:
		- baseline (no filtering)
		- with filter
	- 3 Non-Differential Functions

## ================================================================================================
## Email:

Πρόσβαση στον server για διπλωματική εργασία

Προς: parapi@tuc.gr
, ngiatrakos@tuc.gr

Καλησπέρα σας,

Ονομάζομαι Αθανάσιος Χριστόπουλος (ΑΜ: 2022030077) και είμαι φοιτητής του ΗΜΜΥ. Στο πλαίσιο της διπλωματικής μου εργασίας, με επιβλέποντα καθηγητή τον κ. Γιατράκο, θα χρειαστώ πρόσβαση στον server, καθώς το project που υλοποιώ απαιτεί αυξημένους υπολογιστικούς πόρους (κυρίως GPU και μνήμη).

Εφόσον είναι εφικτό να μου δοθεί πρόσβαση, θα ήθελα επίσης να ενημερωθώ για τυχόν κανόνες σχετικά με τη χρήση του server. Για παράδειγμα, φοβαμαι οτι θα χρειαστώ σημαντικό χώρο αποθήκευσης (κατά προσέγγιση ~30 GB, δεν ξερω αμα ειναι προβλημα ή οχι).

Ευχαριστω πολυ,
Αθανάσιος Χριστόπουλος

uname -m 
nvidia-smi
ldconfig -p | grep libcudart
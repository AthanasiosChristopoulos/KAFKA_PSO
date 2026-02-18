
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
		Limit communication στο pBest_weights_topic pipeline:
			-> Στειλε μονο οταν το loss εχει αλλαξει significantly 

		- Αλλαζε το κατωφλι T του significant loss δυναμικα, στην αρχη να ειναι μικρο και μετα να αυξανεται.

		Geometric Monitoring:
			- Actuall GM:
				- Synchronization necessary
			- παραλλαγης στο gm:
				- ενας worker i βλεπει οτι κανει cross το border / to A / φευγει απο το κατωφλι
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
			- το sync στο PSO ενδεχεται να ειναι αντιφατικο, καθως στο PSO περιμενουμε το "sync" να ερθει ως αποτελεσμα του swarm να κανει converge, 
					οχι να γινει enforced.
				- Για να γινει αυτο τα particles κανουν maintain το δικο τους position, δεν κανουν sync
		
		Limit communication στο local_weights_topic pipeline:
			- οταν η current κατασταση του particle θα αποτελεσει μια σημαντικη αλλαγη στο FedAvg / Global Model, μονο τοτε στειλε 
			- Federated Learning component of your project, averaging (FedAvg) and καθε ποσο κανω track το position
				- μεχρι τωρα, καθε 60 updates. Θα πρεπει να αλλαξει αυτο, να κρινεται απο το ποσο εχει κανει diverge
				- Measure of distance με προηγουμενη κατασταση (αμα εχει αλλαξει significantly ή κατασταση)

		Prediction models σημαινει προβλεψη pbest / gbest, ωστε να ξερω αμα θα επικοινωνησω ή οχι


	2) Experimentation - Setup:
		- Different models
		- Different Non - Differentiable Loss Functions
		- Different Datasets, Different sizes: INDEPENDENT_WORKER_DATA_PROCESSING true or false
		- Measure performance and latencies
		- Experimentation Setup:
			- Hardware (Laptop / Server (Cloud) / Raspberry Pis)
				- Actually mention / analyse the specs
				- Run on both highend and lowend devices 
			- Use and test as many different:
				- model / method parameters as possible
				- measurement methods (error definitions) 
			
		- Measure / Diagrams of:
			- Number of workers => increases parallelization / speed (throughput) - how much data the pipeline can process, bytes (records) per second
			- how much time on average does it take you to process one record 
				- this is supposed to be a streaming application: input_mbps < process_mbps
			- x: Number of training samples - y: loss / accuracy / F1 score	
			- Limit communication:
				- Run without => identify latency
				- Run with	  => makes training more efficient / faster
			- This is affected by Model Size (Number of Weights - NN?K) / Number of workers / TRAIN_SIZE / FULLY_INFORMED (PROTOCOL used) / Loss Function:
				- How much time until reached DESIRED_ACCURACY (Training Time) ?
				- How much data until reached DESIRED_ACCURACY - How many epochs ?
				- Αμα το αφησεις να παει οσο παει, τοτε πιο ειναι το ελαχιστο loss / μεγιστο Accuracy που μπορει να φτασει ?
			- Performance and Accuracy Comparison with Gradient Descent
		
	3) Learn more about Kafka Streams + Tensorflow + Neural Learning
		- Improve code on Apache Kafka + Tensorflow:
			- Parallelization (Threading / Instances):
				- On Kafka Streams: Threading and more Kafka Streams instances
				- Kafka Broker: More brokers
				- Parallelize Inference (using an additional Kafka Streams Instance)
				- How many workers / Kafka Streams instances per particle

	6) Related Work + Research:
		- Other attempts implementing PSO
		- how does yours differ ?
			- Implementation / Enviroment / Technologies (Docker, Java, Kafka Streams)
			- What scientific differences did you add / remove / what parameters did you use ?
	
	19) velocity initialiazation (magnitude)
			
	22) Do the diagrams / experimentation on (Measurement is taken when all training data has been consumed):
		- Evaluate multiple workers scenario (x_g performance gets worse ?)
		- Accuracy - N workers (αυξουσα σχεση)
		- Training Time - N workers	(φθινουσα σχεση)
		- Communication is more costly as the size of the model increases

	26) CNNs
		
	30) Ευρεση καλυτερων Datasets (πιο δυσκολα) - MLP datasets που actually χρειαζονται μεγαλυτερα μοντελλα
		=> Maybe non IID Datasets

	38) Add regularization manually after .output for weights

	39) Different Non - Differentiable Loss Functions. You need to find about 3 such functions and show that they are non differentiable

	40) You need to measure exactly how much communication costs time wise (like you did for Kafka record processing / foward passes)
		- communication here isnt data record reading, but weightmessges exchanged between coordinator and worker

	43) PSO Tranfer Learning:
		- Ευρεση καταλληλου base model for MNIST and CIFAR + trainable End Layers
		- Θα πρεπει να βρεις additional Layers + Non Differentiable Loss Functions, ωστε:
			- το base model να μην δουλευει καλα
			- να κανεις train το frozen base model  το trainable End Layers, ωστε να δουλευει καλυτερα απο το σκετο base model
			- συνηθως δεν πας να κανεις train from scratch

	47) Κανε share one drive με γραφικες (διαγραμματα και τετοια ...)

## =======================================================================================

	Backlog Tasks:
	
	1) ? GUI for:
		- enviroment variables configuration 		
		- monitoring training progress
		- inference
		- Finding a third party library GUI for defining a neural model architecture ?
	
	21) Implement neighbourhood topologies more efficiently:
		=> Problem: With current protocols, if there are N Workers, there are N neighborhoods
		=> Change Kafka / Kafka Streams architecture with Routers to fascilitate neighborhoods
	13) Figure out Kafka Streams aggregation HOF thing (straight from Kafka Streams DSL)

	11) Add buffer as a statestore (on worker who is accumulating samples)

	17) X_G state store - the global current position

	20) Find non IID datasets - Partitioning of Non IID data if possible (this depends on the dataset):
		- Up to now just Round Robin
		- Otherwise you need to do it with keyed Records

	21) Use Vector class not float[] => important for calculations

	29) Have both training and test accuracy => really hard on the coordinator because it relies on loading the entire test dataset.
	
	((37) Dynamic Enviroments / Dynamic PSO (there is a specific paper for this)))

	41) Συγκριση η τελικη θα γινει με κατανεμημενο περιβαλλον στο training του gradient descent:
		- οχι centralized, αλλα gradient descent. 
		- Δηλαδη θελουμε να συγκρινουμε PSO vs gradient descent σε distributed περιβαλλον
		- Δες parameter server (independent learning με merging των μοντελλων afterwards) => δεδομενα ειναι λιγοτερα για καθε worker
		- Basically πρεπει να κανεις το federated learning στην python με gradient descent

	42) Partition wizard stuff για να δωσεις περισσοτερο χωρο στο Ubuntu

	46) Fix nonexistent error logging. Set Index to wrong (HEAD_LAYER_IDX = 9)

## =====================================================================================================
	Thesis Structure:
		- Motivation, Table of Contents
		- Related Work / Thesis Contribution
		- Introduction (Theoretical Background)
		
		- Implementation:
			- Architecture (Kafka Topology, Project Class Hierarchy)
			- Protocol (Differences between normal PSO protocol and distributed / Federated learning / Kafka PSO protocol)
				=> many things, like the sharing and updating of gBest, can be easily done on the same device.
				=> that isnt the case with federated, communication is costly 
			- Functional and Non Functional Requirements
			
		- Experimental Evaluation:
			- Run it locally 
			- Run it on a server
			
		- Future Work

## ================================================================================================
## Email:

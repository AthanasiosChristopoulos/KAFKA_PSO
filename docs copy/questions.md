
Ερωτησεις για Διπλωματικη:

	- Showcase: ==========================================================================================
	
	- Experimentation: =============================================================

		- σκεφτηκα για GM οτι μπορω να το χρησιμοποησω μονο για monitoring του Federated Average Model χωρις full synchronization, μονο 
			partial synchronization of one worker, αλλα θα πρεπει να χρησιμοποιησω το FFT για dimensionality reduction και δεν ξερω αμα αξιζει
			- this is a crazy amount of work for a Kafka Streams processor (we need to do this per record)
			- Geometric monitoring is actively disfunctional for A) type filtering. Even if the difference is large, this is expected, we want exploration 
			in the start and convergence in the end. This means that for monitoring purposes, the difference between the distance of the models is irrelevant. We want to monitor more during the end, not when they have a significant difference
				=> this isnt federated learning, we dont want to know when to sync just when to monitor

		- αυτην την στιγμη το communication cost ειναι μικροτερο απο οτι θα ειναι στην πραγματικοτητα ? Δηλαδη σε ενα πραγματικο Federated Enviroment, με remote communication protocol, to communication θα ειναι πιο costly ? Γιατι αυτην την στιγμη το processing of data + forward pass ειναι το bottleneck.
			=> Async protocol, αρα δεν παιζει
						
		- Pretrained ImageNet Models from the Internet on independent datasets wont work. I need to either train my own models and then leave last layer only for training ? This layer would then be "fine-tured" to work with a "non differentiable" loss function or i can get models from the internet that are more specific to our problem (same dimensionality same dataset trained on). Is there a preference between these options ? 
			Is it ok if:
				=> Many epochs
				=> Long Time (because of forward pass)
				=> My pretrained model on the same dataset removing only last layer maybe more
					=> or a lightweight model of the same size, but of a different / more generall dataset
				=> Probably need to run it on the server

		- Memory consumption idea ?

		- Μπορω να αρχισω να γραφω ?
		
		- Federated Learning or Distributed Learning ?
			=> Αρκει να αλλαξω το dataset σε NON-IDD ?
			=> Αρκει οτι το συστημα εχει μονο το capacity να το κανει αυτο και οτι απλως δεν το εξεταζουμε / δεν το τρεχουμε ετσι για διευκολυνση ?

		- Τα results και overall experimentation να τα τρεξω για server ή οχι ? ποια να τρεξω σε server και πια οχι ? Μονο αυτα που χρειαζονται ?


	- Για συνταξη της Διπλωματικης + Next Steps: ===========================================

		- Ειναι τα Tasks που εχω σκεφτει σωστα και ξεχναω κατι ?:
			- Comparison with Gradient Descent ?
			- Μετρησεις προς το Experimentation Setup (τι κυματομορφες) ?	- Μετρησεις προς το Experimentation Setup (τι κυματομορφες) ?nv as User Interface ?
			- Μετρησεις προς το Experimentation Setup (τι κυματομορφες) ?
			
		- εχω βρει αυτα τα papers να βαλω στην διπλωματικη

		- για τιτλο μηπως λειπει οτι χρησιμοποιουμε το PSO για training neural networks ? Η εννοειται ? 	
			- Σε αλλα papers χρησιμοποιουν PSO γενικα ως τροπο βελτιοστοποιησεις μιας συναρτησεις

	- Implementation Detail: ================================================================

		- velocity like simulated annealing ? Make velocity reduce over time so that swarm converges ... 
		- Χρησιμοποιω στον κωδικα καποια "κολπα" για να δουλευει το Kafka Streams με τους brokers πειραζει ?
		- τρεξιμο σε server reliance on static classes για coordination

	- Email: ==========================================================================


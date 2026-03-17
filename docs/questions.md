
Ερωτησεις για Διπλωματικη:

	- Showcase: ==========================================================================================
	
	- Experimentation: =============================================================

		- σκεφτηκα για GM οτι μπορω να το χρησιμοποησω μονο για monitoring του Federated Average Model χωρις full synchronization, μονο 
			partial synchronization of one worker, αλλα θα πρεπει να χρησιμοποιησω το FFT για dimensionality reduction και δεν ξερω αμα αξιζει
			- this is a crazy amount of work for a Kafka Streams processor (we need to do this per record)
			- Geometric monitoring is actively disfunctional for A) type filtering. Even if the difference is large, this is expected, we want exploration 
			in the start and convergence in the end. This means that for monitoring purposes, the difference between the distance of the models is irrelevant. We want to monitor more during the end, not when they have a significant difference
				=> this isnt federated learning, we dont want to know when to sync just when to monitor
								
		- Τι εννουσαι με scaling σε πιο δυσκολα Datasets
			-> ειναι οκ να κανω incrementally sacrifice some stuff (ποιοτητα λειτουργιας του PSO)
				=> Many epochs
				=> Long Time (because of forward pass)
				=> My pretrained model on the same dataset removing only last layer maybe more
					=> or a lightweight model of the same size, but of a different / more generall dataset
		
		- svhn -> mnist, οκ ?

		- prediction models ?
		
		- πειραζει που δεν πεφτει το accuracy οταν τρεχω heavy ? Γενικα ειναι απλως πολυ καλυτερο με heavy ... 

		- Προτεινω να βγαλουμε iris και να βαλουμε MNIST5
		
	========================================================================================
	- Για συνταξη της Διπλωματικης + Next Steps: 

		- εχω βρει αυτα τα papers να βαλω στην διπλωματικη

		- για τιτλο μηπως λειπει οτι χρησιμοποιουμε το PSO για training neural networks ? Η εννοειται ? 	
			- Σε αλλα papers χρησιμοποιουν PSO γενικα ως τροπο βελτιοστοποιησεις μιας συναρτησεις

		- το related work section ? ειναι για papers που κανουν κατι παρομοιο ? οχι απλως γενικα στο ιδιο field of study ?

		- Πως ακριβως να βλεπω το experimentation ?
			- Κανουμε study each individuall dataset by itself σιγα σιγα ?
			- Δηλαδη κανουμε για ενα Dataset:
				- Diagram: N_WORKERS => Ευρεση SWEET SPOT N_WORKERS για αυτο το DATASET
				- Diagram: STRENGTH => Ευρεση SWEET SPOT SEVERIY για αυτο το DATASET
				- Final Diagram, using the above determined parameters, to achieve best accuracy scenario:
					=> accuracy - iterations
			- We need to do a more generall study first:
				- do it on pendigits: using pendigits as generall example (with INDEPENDENT = false / true)
				to explain generally and then look at each dataset individually all diagrams
				
		- Στο Related Work μπαινουν sources / papers που κανω citate στην δικη μου δουλεια ? Η εχει πιο γενικη φυση ?
		Το Related Work είναι πιο γενικής φύσης, αλλά συνήθως περιλαμβάνει papers που κάνεις cite στη δουλειά σου.

		- Εχει νοημα να βαλω ενα Model Definition section μπας και εξηγησω το Transfer Learning + DL4J code

		- Πως ακριβως να βλεπω το experimentation ?
			- Κανουμε study each individuall dataset by itself σιγα σιγα ?
			- Δηλαδη κανουμε για ενα Dataset:
				- Diagram: N_WORKERS => Ευρεση SWEET SPOT N_WORKERS για αυτο το DATASET
				- Diagram: STRENGTH => Ευρεση SWEET SPOT SEVERIY για αυτο το DATASET
				- Final Diagram, using the above determined parameters, to achieve best accuracy scenario:
					=> accuracy - iterations
			- Keep the generall study using pendigits as generall example (with INDEPENDENT = false / true)
				to explain generally and then look at each dataset individually all diagrams

	- Email: ==========================================================================

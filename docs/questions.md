
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

		========================================================================================

		- Ρωτα τον για παρουσιαση
			- Οτι εχω κανει οπως ειναι στο .pdf, να το περασω σε powerpoint
			- Να κανω omit κατι σαν το Background ???
			- 19/5/2026 ή 19/6/2026
			- θα μιλαω σε greeklish ???
			
		- οχι SUSY , χρειαζεται ? να βαλω κατι αλλο instead ?
		
		- Comments on comments / clarifications:
			- may decrease. it is not that they always decrease. right?
			- but does allow distributed PSO reducing the execution time of the centralized one. Recall that PSO is, in turn, necessary for training over non-differentiable loss functions where classic SGD-like training is inapplicable. 

		- Ποια ειναι αυτα τα 1 - 2 πραγματα ακομα ... ? 
			- hopefully not improvement in accuracy ή prediction models
			- υπαρχουν γενικα areas βελτιωσεις ? ΠΟυ θα εχουν καποιο αποτελεσμα
			
		- Κανε την Παρουσιαση

		- Βαθμολογικο Προβλημα
		- Μηπως θα υπηρχε καποια βοηθεια για το job market ή καποιο research position
			- Ειναι εξυπνο να αφησεις αυτην την πορτα ανοιχτη ? ισως ?
	
		- but does allow distributed PSO reducing the execution time of the centralized one. Recall that PSO is, in turn, necessary for training over non-differentiable loss functions where classic SGD-like training is inapplicable. 

		- Αμα ειναι μπορουμε να σβησουμε απο Background κ ?

		- Θα μου δειτε τον κωδικα, θα πρεπει να τον καθαρησω ?

		- Να βαλω νεο Neighborhood config ... ? Να κανω ενα N^2 + N στα διαγραμματα ?

	========================================================================================
	========================================================================================
	========================================================================================

	- Για συνταξη της Διπλωματικης + Next Steps: 

		- για τιτλο μηπως λειπει οτι χρησιμοποιουμε το PSO για training neural networks ? Η εννοειται ? 	
			- Σε αλλα papers χρησιμοποιουν PSO γενικα ως τροπο βελτιοστοποιησεις μιας συναρτησεις

		- Εχει νοημα να βαλω ενα Model Definition section μπας και εξηγησω το Transfer Learning + DL4J code

	- Email: ==========================================================================


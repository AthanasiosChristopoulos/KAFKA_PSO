
Ερωτησεις για Διπλωματικη:

	- Showcase: ==========================================================================================

		- Running in Ubuntu enviroment

		- Serialized from JSON messages to bytes
		- Partitioning - Data Parallelism: χωριζω τα δεδομενα ανα worker (N workers) αλλα αυξανω συνολικα τα δεδομενα κατα Ν

		- Test Graphs on (changed protocol to be training first to do the graphs - training time. Is training first correct ?):
			- between Fully Informed vs Neighborhood best 
			- Limitation / Filter or no limitation in the communication
			- between differentiable and non-differentiable loss functions 
			- δεν μπορω να αυξησω τον παραλλελισμο παραπανω, αφου το pc μου εχει limited number of cores / true parallelism:
				- N_WORKERS => [2, 4, 6] or [3, 5, 8]

		- New Datasets
			=> Susy runs with 0.77 but on GD i am getting 0.8 cant get better in GD these results on PSO are fine.
				=> Maybe something wrong with the model
			=> Pendigits is easier to see improvement since multiple classes

		- Cacherd data set only 500 samples always used

	- Experimentation: ==========================================================================================

		- σκεφτηκα για GM οτι μπορω να το χρησιμοποησω μονο για monitoring του Federated Average Model χωρις full synchronization, μονο 
			partial synchronization of one worker, αλλα θα πρεπει να χρησιμοποιησω το FFT για dimensionality reduction και δεν ξερω αμα αξιζει
			- this is a crazy amount of work for a Kafka Streams processor (we need to do this per record)
			- Geometric monitoring is actively disfunctional for A) type filtering. Even if the difference is large, this is expected, we want exploration 
			in the start and convergence in the end. This means that for monitoring purposes, the difference between the distance of the models is irrelevant. We want to monitor more during the end, not when they have a significant difference
				=> this isnt federated learning, we dont want to know when to sync just when to monitor

		- αυτην την στιγμη το communication cost ειναι μικροτερο απο οτι θα ειναι στην πραγματικοτητα ? Δηλαδη σε ενα πραγματικο Federated Enviroment,
			με remote communication protocol, to communication θα ειναι πιο costly ? Γιατι αυτην την στιγμη το processing of data + forward pass ειναι 
			το bottleneck.
				=> Async protocol, αρα δεν παιζει
						
		- Σωστα εκανα τα non differential functions ? Το λεω γιατι ειναι εκτος της ιδια της συναρτησης δεν επιρεαζουν directly την loss συναρτηση
		
		- Pretrained ImageNet Models from the Internet on independent datasets wont work. I need to either train my own mdels and then leave last 
			layer to be trained "non differentiably" or i can get models rom the internet that are more specific to our problem (same dimensionality 
			=> same dataset trained on). Is there a preference between these options ? 


		
	- Για συνταξη της Διπλωματικης + Next Steps: ==========================================================================================

		- Ειναι τα Tasks που εχω σκεφτει σωστα και ξεχναω κατι ?:
			- Comparison with Gradient Descent ?
			- GUI / .env as User Interface ?
			- Μετρησεις προς το Experimentation Setup (τι κυματομορφες) ?
			
		- εχω βρει αυτα τα papers να βαλω στην διπλωματικη

		- για τιτλο μηπως λειπει οτι χρησιμοποιουμε το PSO για training neural networks ? Η εννοειται ? 	
			- Σε αλλα papers χρησιμοποιουν PSO γενικα ως τροπο βελτιοστοποιησεις μις συναρτησεις

	- Implementation Detail ===================================================================================================

		- velocity like simulated annealing ? Make velocity reduce over time so that swarm converges ... 
		- Χρησιμοποιω στον κωδικα καποια "κολπα" για να δουλευει το Kafka Streams με τους brokers πειραζει ?
		- τρεξιμο σε server reliance on static classes για coordination

	- Email: ===================================================================================================

	Καλησπερα,
	Εχω κανει τις περισσοτερες βελτιωσεις που συζητησαμε κατα την προηγουμενη μας συναντηση για το PSO project (ειδικα τα τα διαγραμματα με 
	Number of Workers - Training Time - Accuracy με ή χωρις filtering ειναι λιγο ασταθης, αλλα πανε καλα νομιζω) και επισης εχω βρει δυο datasets 
	που με βοηθανε περισσοτερο στο experimentation.

	Το προβλημα στο οποιο εχω κολλησει ειναι το Training των CNNs, και γενικοτερα η ικανοτητα του 
	PSO να αποδοσει (απο αποψη accuracy) σε πιο δυσκολα datasets.

	Στην περιπτωση του CNN, βλεπω οτι οσο και να απλοποιω το προβλημα (δηλαδη αρχισα με CIFAR, και μετα το πηγα στο MNIST dataset με μονο 4 κλασσεις),
	δεν βλεπω σχεδον καθολου βελτιωση στο Accuracy. Νομιζω οτι το PSO δεν ειναι καταλληλο για να κανει train ενα CNN, επειδη σε συγκριση με ενα απλo
	Dense Neural Network, εχει πολυ μεγαλο κοστος forward pass (το PSO κανει πολυ περισσοτερα forward passes απο gradient descent) και ταυτοχρονα ειναι πολυ
	πιο sensitive σε αλλαγες παραμετρων. Εκτος απο αυτο το training ενος CNN ειναι ηδη δυσκολο για gradient descent (απο αποψη χρονου και accuracy και αναγκη απο 
	πολλα epochs), αρα ειναι πολυ πιο δυσκολο για PSO.

	Θα ηθελα να ρωτησω αμα μπορουμε να επικεντρωθουμε σε απλα Dense Neural Networks ή αμα ειναι σημαντικο να δουλευει το PSO με CNNs ? Σε αλλα papers παρατηρω 
	κυριως ενα hybrid approach (δηλαδη πρωτα gradient descent και μετα PSO), αλλα αμα καταλαβαινω σωστα αυτο δεν επιτρεπεται εδω επειδη πρεπει να 
	χρησιμοποιουμε non differentiable Loss Functions.

	Ευχαριστω πολυ, 
	Αθανασιος Χριστοπουλος


Καλησπέρα,

Έχω κάνει τις περισσότερες από τις βελτιώσεις που συζητήσαμε στην προηγούμενή μας συνάντηση για το PSO project, ειδικά όσον αφορά τα διαγράμματα (Number of Workers – Training Time – Accuracy), με ή χωρίς filtering, τα αποτελέσματα είναι λίγο ασταθή, αλλά πηγαίνουν σχετικά καλά. Επίσης, έχω βρει δύο datasets που με βοηθούν περισσότερο στο experimentation.

Το βασικό πρόβλημα στο οποίο έχω κολλήσει είναι το training των CNNs, και γενικότερα την ικανότητα του PSO να αποδώσει (accuracy) σε πιο απαιτητικά datasets.

Στην περίπτωση των CNNs παρατηρώ ότι, όσο κι αν απλοποιώ το πρόβλημα (ξεκινώντας από CIFAR και φτάνοντας μέχρι MNIST με μόνο 4 κλάσεις),  δεν βλέπω σχεδόν καμία ουσιαστική βελτίωση στο accuracy. Νομίζω ότι το PSO δεν είναι ιδιαίτερα κατάλληλο για training CNNs, κυρίως επειδή, σε σχέση με ένα απλό Dense Neural Network, το CNN έχει πολύ μεγαλύτερο κόστος στο forward pass και το CNN είναι  πιο ευαίσθητο σε αλλαγές παραμέτρων.
Γενικότερα το training CNNs είναι ήδη απαιτητικό ακόμη και με gradient descent, άρα γίνεται ακόμη δυσκολότερο σε PSO.

Θα ήθελα να ρωτήσω αν θα μπορούσαμε να επικεντρωθούμε σε πιο απλά Dense Neural Networks ή αν είναι σημαντικό το PSO να λειτουργεί και σε CNNs. 
Σε papers που έχω δει χρησιμοποιούνται κυρίως hybrid approaches (π.χ. αρχικά gradient descent και στη συνέχεια PSO),  αλλά αν καταλαβαίνω σωστά αυτό δεν επιτρέπεται εδώ, καθώς απαιτείται η χρήση non-differentiable loss functions.

Ευχαριστώ πολύ,
Αθανάσιος Χριστόπουλος

## ================================================================================================
## Papers I have already downloaded:

1)
Zhang, Zhang, Lok, Lyu (2007), A hybrid particle swarm optimization–back-propagation algorithm for feedforward neural network training. => The paper explicitly says it combines PSO global search with BP local search to train network weights, and it uses a heuristic transition from PSO to gradient-based search.

2)

de Rosa, Roder, Papa, dos Santos (2022), Improving Pre-Trained Weights Through Meta-Heuristics Fine-Tuning.
	=> PSO fine-tuning not actuall training

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
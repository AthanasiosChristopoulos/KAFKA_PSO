
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
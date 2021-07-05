Genetic Algorithms
---

Genetic algorithms can be used in Python via the tpot package.

You need to define the number of generations, population_size and offspring_size that feeds into the model.
The result will be the most successful algorithm across each of the generations.

```python
from tpot import TPOTClassifier

number_generations = 5  # 5 generations of the algorithm
population_size = 4  # Start with 4 algorithms
offspring_size = 3  # offspring set at 3
scoring_function = 'accuracy'
# set the scoring that you are trying to maximize (be wary of using accuracy as it is not appropriate for unbalanced
# classes

# Create the tpot classifier
tpot_clf = TPOTClassifier(generations=number_generations, population_size=population_size,
                          offspring_size=offspring_size, scoring=scoring_function,
                          verbosity=2, random_state=1, cv=10)

# # Fit the classifier to the training data
tpot_clf.fit(X_train, y_train)

# Best algorithm to fit our data :
# Best pipeline: DecisionTreeClassifier(RBFSampler(input_matrix, gamma=0.75),
# criterion=entropy, max_depth=10, min_samples_leaf=12, min_samples_split=7)

```

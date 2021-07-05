
"""Import packages required"""

import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tpot import TPOTClassifier

"""# Import the iris dataset"""
iris = sns.load_dataset("iris")

"""Typical queries used to evaluate your data - always carry this out before completing any analysis
    on your data"""
iris.head()
iris.info()
iris.describe()
iris.columns
iris.isnull().sum() # there are no null values in the data

"""Check the correlation between each of the vars"""
"""Sepal length and Sepal width as well as petal length and petal with look to be highly correlated"""
sns.pairplot(iris)
iris.corr()

################################################################################################################
                # Random Forest used to predict the species
################################################################################################################

"""# Importing the dataset"""
X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

"""# Splitting the dataset into the Training set and Test set"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

"""# Feature Scaling"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

"""# Training the Random Forest model on the Training data (without tuning the hyperparameters)"""
classifier = RandomForestClassifier(random_state = 1)
classifier.fit(X_train, y_train)

"""# Predicting the Test set results"""
y_pred = classifier.predict(X_test)

"""# Making the Confusion Matrix"""
"""Overall accuracy is 97% (please note F1 score is usually a better indicator of the success of the model
   especially if we have unbalanced classes)
   F1 score is also 97% 
   There was only one case which was misclassified"""
cm = confusion_matrix(y_test, y_pred)
print(cm)
cr = classification_report(y_test, y_pred)
print(cr)
accuracy_score(y_test, y_pred)



#################################################################################################################
                     # Genetic Algorithm code
#################################################################################################################

# Genetic Algorithms

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


"""# Making the Confusion Matrix"""
"""Overall accuracy is 97% (please note F1 score is usually a better indicator of the success of the model
   especially if we have unbalanced classes)
   F1 score is also 97% 
   There was only one case which was misclassified"""
y_ga_pred = tpot_clf.predict(X_test)
cm_rf = confusion_matrix(y_test, y_ga_pred)
print(cm_rf)
print(classification_report(y_test, y_ga_pred))

# Importing the required libraries
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Reading data
iris = datasets.load_iris()

# Preparing data
x = pd.DataFrame(iris.data, columns = iris.feature_names)
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print(x.head(10))

# Training the model
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

# Predicting the scores
y_pred = classifier.predict(x_test)
print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

# Evaluating the model
print('Accuracy', metrics.accuracy_score(y_test, y_pred))

# Visualising the decision tree
fig = plt.figure()
plot_tree(classifier, feature_names = iris.feature_names, class_names=iris.target_names, filled = True, rounded = True)
plt.show()
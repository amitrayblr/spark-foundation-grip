# Importing the required libraries
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Reading data
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns = iris.feature_names)
print(data.head(10))
# Importing the required libraries
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Reading data 
data = pd.read_csv('data/studentscore.csv')
print(data.head(10))

# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

# Preparing data
x = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 

regressor = LinearRegression()  
regressor.fit(x_train, y_train) 

# Plotting regression line for test data
line = regressor.coef_*x + regressor.intercept_
plt.scatter(x, y)
plt.plot(x, line);
plt.title('Regression Curve')
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.show()

# Predicting the scores
y_pred = regressor.predict(x_test) 
print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}))

# Evaluating the model
print('Mean Absolute Error', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error', metrics.mean_squared_error(y_test, y_pred))

# Testing for your own values
hours = input("Enter the number of hours the student has studied for: ")
print('An expected score for the student would be: ', regressor.predict([[float(hours)]]))


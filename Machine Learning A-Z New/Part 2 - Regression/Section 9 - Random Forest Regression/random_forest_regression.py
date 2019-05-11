# Random Forest Regression
# Lecture 77 https://www.udemy.com/machinelearning/learn/lecture/5855120
# basic idea of random forest is you use multiple Decisions Trees make up 
# a forest. This is also called Ensemble. Each decision tree provides a prediction 
# of the dependent variables.The prediction is the average of all the trees.

# check working directory
import os
os.getcwd()
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting Random Forest Regression to the dataset
# step 1 import the class we want
from sklearn.ensemble import RandomForestRegressor
# step 2 make a regressor across our data
# n_estimators is the number of trees; we started with 10
# random_state is just set to 0
regressor1 = RandomForestRegressor(n_estimators = 10, random_state = 0)
# step 3 fit regressor to our model 
regressor1.fit(X, y)

regressor2 = RandomForestRegressor(n_estimators = 100, random_state = 0)
# step 3 fit regressor to our model 
regressor2.fit(X, y)

regressor3 = RandomForestRegressor(n_estimators = 300, random_state = 0)
# step 3 fit regressor to our model 
regressor3.fit(X, y)

# Predicting a new result
y_pred1 = regressor1.predict([[6.5]])
y_pred2 = regressor2.predict([[6.5]])
y_pred3 = regressor3.predict([[6.5]])

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor3.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
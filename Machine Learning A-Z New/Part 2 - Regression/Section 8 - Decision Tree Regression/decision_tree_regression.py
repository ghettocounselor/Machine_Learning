# Decision Tree Regression
# lecture 71 https://www.udemy.com/machinelearning/learn/lecture/5732730
# to see more about splitting data in DTR review Information Entropy
# it make complete sense even if the underlying math is complex
# see some good images in folder

# check working directory
import os
os.getcwd()

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
# X independent variables
X = dataset.iloc[:, 1:2].values
# y dependent variables
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# not using because dataset it very small
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling - first we'll not use this then see if we need it
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting Decision Tree Regression to the dataset
# step 1 import class from library
from sklearn.tree import DecisionTreeRegressor
# step 2 create regressor object
regressor = DecisionTreeRegressor(random_state = 0)
# step 3 fit regressor to our data
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]])

# 1st run of Visualising the Decision Tree Regression results (higher resolution)
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()










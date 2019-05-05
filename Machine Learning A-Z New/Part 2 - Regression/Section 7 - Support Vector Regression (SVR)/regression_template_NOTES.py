# This was an attempt in class Lecture 61 
# here https://www.udemy.com/machinelearning/learn/lecture/5846952
# to construct a TEMPLATE for use in the future. There are some odd assumptions. 
import os
os.getcwd()
# see path of current module (file in use)
import os, sys
os.path.dirname(os.path.abspath(sys.argv[0]))
# set working directory to path of file being used
import os, sys
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# # Importing the dataset
# =============================================================================
dataset = pd.read_csv('Position_Salaries.csv')
# in our data we only need the years (independent) and salary (dependent variable)
# we want a matrix so we'll use Py against itself
# remembering that the upper value is ignored we add it in
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# =============================================================================
# missing data - none
# =============================================================================

# =============================================================================
# Splitting the dataset in to Train and Test
# =============================================================================
# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# =============================================================================
# Encoding - Categorical variables - none
# =============================================================================

# =============================================================================
# Feature Scaling - MOST regression models don't require this step
# =============================================================================
# only needed for times when we need to manually do this, often time not
# here we are just adding another variable so no need to do this here
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# =============================================================================
# Fitting Non-linear Regression to the dataset
# =============================================================================
# create your regressor here

# =============================================================================
# Predicting a new result
# =============================================================================
y_pred = regressor.predict(6.5)

# =============================================================================
# Visualize
# =============================================================================
# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# higher resoltuion
X_grid = np.arange(min(X), max(X), 0.1)  # this is a vector, we want and array
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

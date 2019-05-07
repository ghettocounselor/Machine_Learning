# SVR lecture 69 https://www.udemy.com/machinelearning/learn/lecture/5952220
# book here on SVR https://link.springer.com/book/10.1007/978-1-4302-5990-9
# in SVR we have a slightly different intention than with LR
# in SVR we want to control the thresholds that we set, LR we are just looking for a best fit. 
import os
os.getcwd()
# =============================================================================
# Importing the libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Importing the dataset
# =============================================================================
dataset = pd.read_csv('Position_Salaries.csv')
#  salary Y (dependent) and years X (independent variable)
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# =============================================================================
# Feature Scaling - for SVR we need to feature scale our data
# =============================================================================
# remember that the purpose of feature scaling is to make adjustment for 
# differing units between the variables; in this case years and salary
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

# =============================================================================
# Fitting SVR to the dataset
# =============================================================================
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # kernel is most important variable
# fit regressor to our data
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[6.5]]) # this didn't work
# since we feature scaled, we need to transform our y_pred of 6.5
# we do this by applying the transformation, and because we actually 
# are looking to predict the salary from the years we need to inverse 
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
# OMG https://www.udemy.com/machinelearning/learn/lecture/5952220

# =============================================================================
# outcome - SVR is a good model as it predicted well
# =============================================================================
# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# manually saved from the popup

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# manually saved from the popup
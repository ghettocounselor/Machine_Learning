# Polynomial Regression
# lecture 57 https://www.udemy.com/machinelearning/learn/lecture/5813756
# get working directory Spyder
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
X = dataset.iloc[:, 1].values
# here we have X as a vector, we really want a matrix so we'll use Py against itself
# remembering that the upper value is ignored we add it in
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# =============================================================================
# missing data - none
# =============================================================================

# =============================================================================
# Splitting we don't want to do because the set is so small
# =============================================================================
# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# =============================================================================
# Encoding - Categorical variables - none
# =============================================================================

# =============================================================================
# Feature Scaling - no here as we didn't do this for the SLR or MLR 
# =============================================================================
# here we are just adding another variable so no need to do this here
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# =============================================================================
# Fitting Linear Regression to the dataset
# =============================================================================
# we are making LR here to compare to the PLR
from sklearn.linear_model import LinearRegression
# lets make a model, we'll call it lin_regressor
lin_regressor = LinearRegression()
lin_regressor.fit(X, y)

# =============================================================================
# Fitting Polynomial Regression to the dataset
# =============================================================================
# let's get a new class here PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
# now we'll make a model with polynomial variables xsquared as in a^2
poly_regressor = PolynomialFeatures(degree = 2)
# in video we started with 2 then it changed to 4
# now we apply the model to the data X, aka we are transforming X into X_poly
X_poly = poly_regressor.fit_transform(X)
# NOW! we'll make the polynomial Y coordinates of our data
#poly_regresor.fit(X_poly, y)
lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(X_poly, y)

# altered the degree to 3 we did this to have things drawn closer to the variables we have
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 3)
X_poly = poly_regressor.fit_transform(X)
lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(X_poly, y)

# just for fun let's try 4 - it's predicting well
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4)
X_poly = poly_regressor.fit_transform(X)
lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(X_poly, y)

# =============================================================================
# Visualize
# =============================================================================
# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
# we'll add the predictions for Salary
plt.plot(X, lin_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
# here we need to use our regression built with the model for our 
# polynomial regression. 
plt.plot(X, lin_regressor_2.predict(poly_regressor.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# let's do an imrovement to get less straight line action of our curve between points
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
# what we'll do is adjust X_grid with numpy arange
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_regressor_2.predict(poly_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# =============================================================================
# Predicting a new result with Linear Regression
# =============================================================================
# we can pass the regressor model a value to predict a value
lin_regressor.predict([[6.5]])
# Out[84]: array([330378.78787879])  What! ;) this is correct with SLR

# Predicting a new result with Polynomial Regression
lin_regressor_2.predict(poly_regressor.fit_transform([[6.5]]))
# Out[85]: array([158862.4526516])


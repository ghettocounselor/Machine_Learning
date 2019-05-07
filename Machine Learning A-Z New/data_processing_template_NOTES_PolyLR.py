# Data Preprocessing Template
# =============================================================================
# couple of short cuts
## command 1 for a # 
## command 4 for a block quote like this area
## highlight someting and command I for help on that item hit command I twice
# =============================================================================
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
import numpy as np                 # numpy is a mathmatical tool for python
import matplotlib.pyplot as plt    # ploting library very cool
import pandas as pd                # helps us import data
import pandas_profiling as pf      # cool tool to look at your dataset
"""import sys
!{sys.executable} -m pip install pandas-profiling"""
# check working directory
getwd()

# =============================================================================
# import data and setup Independent X and dependent y variables
# =============================================================================

# using Pandas we'll Import the dataset for the notes file only !!
dataset = pd.read_csv('/Users/markloessi/Machine_Learning/Machine Learning A-Z New/Part 1 - Data Preprocessing/Data.csv')
# using out of the box describe
dataset.describe()
# now pandas profiling
# from here https://towardsdatascience.com/speed-up-your-exploratory-data-analysis-with-pandas-profiling-88b33dc53625
# not working as expected?? spyder
"""
pf.ProfileReport(dataset)"""
# basically next we want to separate the Independent (X) and Dependent (y) variables. TRUTH => Y (dependent) and  X (independent) 
# x will be the array that makes up the independent variable(s)
# reading the [] bit we are saying to the left of the , that we want all the variables
# on the right of the , we are saying all of them except the one on the right (-1)
X = dataset.iloc[:, :-1].values
# y will be the array that makes up the dependent variable
# 0,1,2,3 so 3 is the purchased column and the dependent variable
y = dataset.iloc[:, 3].values

# scatter plot the data
plt.scatter(X,y)

# =============================================================================
# missing data
# =============================================================================

# lecture 15 from missing_data.py
# Taking care of missing data
# rule of thumb to replace missing data calculate the mean of the other variables and use that
from sklearn.preprocessing import Imputer
# use the Imputer tool to find 
# axis : integer, optional (default=0)
# The axis along which to impute.
# If axis=0, then impute along columns.
# If axis=1, then impute along rows.
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# now we need to use the imputer method of fit to fit to the array/matrix 
# that we want to check for the missing value
# remember the upper bound in the 1:3 (the 3) is not included so we are saying check 
# columns 1 and 2
imputer = imputer.fit(X[:, 1:3])
# now we use the method replace/transform to put the last two pieces of code together
X[:, 1:3] = imputer.transform(X[:, 1:3])

# in console lets have a look, it worked!
# =============================================================================
# X
#Out[24]: 
#array([['France', 44.0, 72000.0],
#       ['Spain', 27.0, 48000.0],
#       ['Germany', 30.0, 54000.0],
#       ['Spain', 38.0, 61000.0],
#       ['Germany', 40.0, 63777.77777777778],
#       ['France', 35.0, 58000.0],
#       ['Spain', 38.77777777777778, 52000.0],
#       ['France', 48.0, 79000.0],
#       ['Germany', 50.0, 83000.0],
#       ['France', 37.0, 67000.0]], dtype=object)
# =============================================================================

# =============================================================================
# Encoding - Categorical variables
# =============================================================================

# Encoding categorical data, basically we are creating Dummy Variables
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# we are just calling the country X 
labelencoder_X = LabelEncoder()
# place cursor at end of fit_transform and command i to see help
# country is column 0, the first column
# X
labelencoder_X.fit_transform(X[:, 0])
# the above kicks out Out[25]: array([0, 2, 1, 2, 1, 0, 2, 0, 1, 0])
# where the different countries have been encoded as numbers 0,1,2
# now we want the fitted transformation to become the first column
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# X
# now we have a problem, the column with 0,1,2 will be misunderstood as numbers
# but they are not; 0 is not < 1 and 1 is not < 2 but it looks that way
# this is where we'll use the OneHotEncoder to basically make 3 columns
# one for each country with 1 and 0 in the rows for yes or no ;) 
# [0] says one hot encode column 0 of what we point it at so to speak
# in this case X from above
onehotencoder = OneHotEncoder(categorical_features = [0])
# note the .toarray is adding the transformation back to the array X
X = onehotencoder.fit_transform(X).toarray()
# X

# =============================================================================
# Encoding - Dependent Variable 
# =============================================================================

# we don't need the OneHotEncoder
# we are using y for the dependent variable
# note we don't add back to the array because above we separated the independent
# and the dependent varaibles into X and Y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# =============================================================================
#  split data Train and Test sets and scatterplot
# =============================================================================

# Lecture 18 https://www.udemy.com/machinelearning/learn/lecture/5683430
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# train_test_split - Split arrays or matrices into random train and test subsets
# we'll be seperating both X and y 
# test_size is the amount of data, usually 0.2-0.3 depending on how much data you have
# random_state : int, RandomState instance or None, optional (default=None)
# If int, random_state is the seed used by the random number generator; If RandomState 
# instance, random_state is the random number generator; If None, the random number 
# generator is the RandomState instance used by np.random.
# we'll set the random_state = 0 so we have the same data seperation as the course
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# scatterplot data
plt.plot(X_train,y_train, 'go', label='Train Data')
plt.plot(X_test,y_test, 'b*', label='Test Data')
plt.title('Train and Test data Scatterplot')
plt.xlabel('X axis Independent Variable')
plt.ylabel('Y axis Dependent Variable')
plt.legend(loc='best')  # legend text comes from the plot's label parameter.
plt.show()

# =============================================================================
# Feature Scaling and fitting
# =============================================================================
# not always necessary to do this, sometimes the model will take care of this
# Feature Scaling https://www.udemy.com/machinelearning/learn/lecture/5683432
# because Age and Salary are not on the same scale, this will cause problems
# Euclidean problem :) will cause the Salary to be more dominant in our model
# Test this idea by taking the difference between two Ages and two Salaries
# then take the square of them in XLS = # ^2 number to the power of 2
?StandardScaler
from sklearn.preprocessing import StandardScaler
# now we create an object of the StandardScaler
sc_X = StandardScaler()
# then we call that object and use the function fit_transformation on X_train to
# transform X_train
X_train = sc_X.fit_transform(X_train)
# we won't fit the training set only transform it, not clear in lecture why
# apparently because it's always fit to the train set
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
# in the lecture they didn't do the below, but noted it would be necessary for regression
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

# =============================================================================
# visualizations are cool as well - see other section notes
# =============================================================================


# =============================================================================
# Backwards elimination - building of the model
# =============================================================================
# below done with dataset = pd.read_csv('50_Startups.csv')
# note we'll need to include the a constant because the stats model we are 
# using isn't smart enough to include 
# see image IncludeConstant.png
# see lecture 45 https://www.udemy.com/machinelearning/learn/lecture/5789776
import statsmodels.formula.api as sm
# we'll take our vector X and use numpy to append an array of 1's
# we need a matrix of 50 lines of 1's
# the code below will add the 1's column to the right side of the group
X = np.append(arr = X, values = np.once((50,1)),astype(int),axis=1)
# we want the 1's on the left, in column 0, so we invert the process
# and add X to the column of 1's
# interpretation: use numpy to create an array of 50 1's as integers
# and add X (the values in X) along the axis of 1
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)

# =============================================================================
# mannually working through backwards elimination
# =============================================================================
# see file multiple_linear_regression.py for working through manually the elim process
# here: /Users/markloessi/Machine_Learning/Machine Learning A-Z New/Part 2 - Regression/Section 5 - Multiple Linear Regression

# =============================================================================
# very slick automation of backwards elimnination Lecture 48
# =============================================================================
# code in this folder: 
# /Users/markloessi/Machine_Learning/Machine Learning A-Z New/Part 2 - Regression/Section 5 - Multiple Linear Regression
# this file: BackwardsEliminationAutomation.py

# =============================================================================
# Polynomial pieces here
# =============================================================================

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


















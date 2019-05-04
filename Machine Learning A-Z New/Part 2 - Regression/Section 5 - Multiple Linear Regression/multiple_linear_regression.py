# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# get working directory Spyder
import os
os.getcwd()

# =============================================================================
# import data and setup Independent X and dependent y variables
# =============================================================================
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
# indexs in Python start with 0 so our columns are 0,1,2,3,4
# X is independent variables, in this case we say give us everything but the last 
# variable on the right so -1
X = dataset.iloc[:, :-1].values
# the dependent variable is very last on the right so 4th column
y = dataset.iloc[:, 4].values
# to look at the vectors we just created

# scatter plot the data
plt.scatter(X,y) # ValueError: x and y must be the same size

# =============================================================================
# missing data = none this time - we did have some zeros here and there
# =============================================================================


# =============================================================================
# Categorical variables - Encoding
# =============================================================================

# Encoding categorical data - STATES!
# we'll encode our categorical independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
# 3 because our variable is in the column 3 the index 3
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X
# comparing the dummy varialbes we determine
# column 0 is CA, 1 is FL and 2 is NY
# Avoiding the Dummy Variable Trap
# this removes the 0 column the 1st column
X = X[:, 1:]

# =============================================================================
# Splitting the dataset into the Training set and Test set
# =============================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# Feature Scaling - not needed for multiple linear regression the library will do it
# =============================================================================

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# =============================================================================
# Fitting
# =============================================================================
# same library as for simple linear regression
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# and now we 'fit' the model to X and y Training data
regressor.fit(X_train, y_train)

# Predicting the Test set results by applying the regressor model to your X 
# (independent variables) in our X_test to predict Profits
y_pred = regressor.predict(X_test)

# =============================================================================
# there are no visuals for this model as there are too many variables at 5 :)
# =============================================================================
# but we can look at predictions vs actual 
# scatterplot data - !!!!!!! not really working :) although I think it could
plt.plot(X_train,y_train, 'go', label='Train Data')
plt.plot(X_test,y_pred, 'b*', label='Test Data')
plt.title('Train and Test data Scatterplot')
plt.xlabel('X axis Independent Variables')
plt.ylabel('Y axis Dependent Variable')

# =============================================================================
# Backwards elimination - building of the model
# =============================================================================
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
# we're creating the 'intercept' line for our model
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)

# let's create an optimal matrix, as in the variables that have high impact 
# on the dependent variables. We'll do a backwards elimination. 
# X_opt for optimal We'll start with all the variables. 
X_opt = X[:, [0,1,2,3,4,5]]

# step 1 define significance level, we'll use 0.05
# step 2 fit full model with all possible variables
## we'll build a new regressor using ordinary least squares
## we are 'fitting' OLS to the endog and exog varialbes
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
# step 3 consider highest P 
## use summary function of sm to do this
regressor_OLS .summary()
# push results out to CSV for storing
f = open('X_opt_1stelim.csv','w')
f.write(regressor_OLS .summary().as_csv())
f.close()
## x2 is 0.990 so we will remove
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS .summary()
f = open('X_opt_2ndelim.csv','w')
f.write(regressor_OLS .summary().as_csv())
f.close()
## x1 is 0.94
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS .summary()
f = open('X_opt_3rdelim.csv','w')
f.write(regressor_OLS .summary().as_csv())
f.close()
## x2 (which is now 4 in our array) needs to come out as it is 0.602
## we can check the original X to know what x1,x2 are and we can use
## the numbers in our [] below to check them. 
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS .summary()
f = open('X_opt_4thelim.csv','w')
f.write(regressor_OLS .summary().as_csv())
f.close()
## x2 is very close but we will remove x2 as it is at 0.06
X_opt = X[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS .summary()
f = open('X_opt_5thelim.csv','w')
f.write(regressor_OLS .summary().as_csv())
f.close()
## RESULT - R&D spend is the only variable that has an impact
# extra code from a question in lecture 48


































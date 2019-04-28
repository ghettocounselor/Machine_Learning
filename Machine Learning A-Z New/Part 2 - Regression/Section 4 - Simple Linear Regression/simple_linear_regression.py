# Simple Linear Regression - Salary_Data.csv
# get current working directory Spyder
import os
os.getcwd()

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# import data and setup Independent X and dependent y variables
# =============================================================================

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# x will be the array that makes up the independent variable(s) years of experience
# reading the [] bit we are saying to the left of the , that we want all the variables
# on the right of the , we are saying all of them except the one on the right (-1)
X = dataset.iloc[:, :-1].values
# y will be the array that makes up the dependent variable in this case Salary
# 0,1,2,3 so 1 is the Salary column and the dependent variable
y = dataset.iloc[:, 1].values

# scatterplot of data
plt.scatter(X,y)

# =============================================================================
# missing data - - NO MISSING DATA THIS TIME
# =============================================================================

# =============================================================================
#  split data Train and Test sets and scatterplot
# =============================================================================

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
# scatterplot of Train and Test data
plt.plot(X_train,y_train, 'go', label='Train Data')
plt.plot(X_test,y_test, 'b*', label='Test Data')
plt.title('Train and Test data Scatterplot')
plt.xlabel('X axis Independent Variable')
plt.ylabel('Y axis Dependent Variable')
plt.legend(loc='best')  # legend text comes from the plot's label parameter.
plt.show()

# =============================================================================
# Feature Scaling - not necessary for this model, the variables are OK without
# =============================================================================
# also the linear model function will take care of this for us

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# =============================================================================
# Fitting our Model to the Training set 
# =============================================================================

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
?LinearRegression
# now we create an object of the LinearRegression test
# regressor is our 'Machine' in the idea of Machine Learning
regressor = LinearRegression()
# then apply the function 'fit' on X and y_train, interestingly all at the same time
# this is slightly different than fit_transform
# here we 'train' our 'Machine' on our data, the 'Machine' 'Learned'
regressor.fit(X_train, y_train)

# =============================================================================
# Predicting the Test set results
# =============================================================================
# vector of predictions for the dependent variables
# again we use the regressor object that we trained above
# place cursor at end of predict and command i to see help
# in this case we used out Machine regressor to predict salaries from the Texst data
y_pred = regressor.predict(X_test)

# now we'll see how well we did
# Visualising the Train set results against the model's prediction
plt.scatter(X_train, y_train, color = 'red', label='Train Data')
# regression line of the predictions of the Train set of data 
plt.plot(X_train, regressor.predict(X_train), color = 'blue', label='Regression of X_train')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(loc='best')
plt.show()

# Visualising the Test set results against the models predictions (against Train)
plt.scatter(X_test, y_test, color = 'red', label='Test Data')
# regression line
plt.plot(X_train, regressor.predict(X_train), color = 'blue', label='Regression of X_train')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red', label='Train Data')
plt.scatter(X_test,y_test, color = 'blue', label='Test Data' )
# regression line 
plt.plot(X_train, regressor.predict(X_train), color = 'green', label='Regression of X_train')
plt.title('Salary vs Experience (Training & Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(loc='best')
plt.show()

# Data Preprocessing Template
# =============================================================================
# couple of short cuts
## command 1 for a # 
## command 4 for a block quote like this area
## highlight someting and command I for help on that item hit command I twice
# =============================================================================

# Importing the libraries
import numpy as np                 # numpy is a mathmatical tool for python
import matplotlib.pyplot as plt    # ploting library very cool
import pandas as pd                # helps us import data

# check working directory
getwd()
# using Pandas we'll Import the dataset
dataset = pd.read_csv('Data.csv')
# x will be the array that makes up the independent variables
# reading the [] bit we are saying to the left of the , that we want all the variables TRUTH => Y (dependent) and  X (independent) 
# on the right of the , we are saying all of them except the one on the right (-1)
X = dataset.iloc[:, :-1].values
# y will be the array that makes up the dependent variable
# 0,1,2,3 so 3 is the purchased column and the dependent variable
y = dataset.iloc[:, 3].values

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

# Encoding categorical data, basically we are creating Dummy Variables
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# we are just calling the country X 
labelencoder_X = LabelEncoder()
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

# Encoding the Dependent Variable 
# we don't need the OneHotEncoder
# we are using y for the dependent variable
# note we don't add back to the array because above we separated the independent
# and the dependent varaibles into X and Y
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

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

# Feature Scaling https://www.udemy.com/machinelearning/learn/lecture/5683432
# because Age and Salary are not on the same scale, this will cause problems
# Euclidean problem :) will cause the Salary to be more dominant in our model
# Test this idea by taking the difference between two Ages and two Salaries
# then take the square of them in XLS = # ^2 number to the power of 2
?StandardScaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# we won't fit the training set only transform it, not clear in lecture why
# apparently because it's always fit to the train set
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
# in the lecture they didn't do the below, but noted it would be necessary for regression
y_train = sc_y.fit_transform(y_train.reshape(-1,1))


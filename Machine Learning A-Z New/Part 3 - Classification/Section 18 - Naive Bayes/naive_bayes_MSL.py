# Naive Bayes - is an algorithm used in a popular model
# Explained across a couple of lectures, pretty striaght forward idea
# Lecture 115 https://www.udemy.com/machinelearning/learn/lecture/5732736 
# Lecture 116 https://www.udemy.com/machinelearning/learn/lecture/5732738
# Lecture 117 https://www.udemy.com/machinelearning/learn/lecture/5732740
# Lecture 118 https://www.udemy.com/machinelearning/learn/lecture/5732742
# Python Lecture 119 https://www.udemy.com/machinelearning/learn/lecture/5740200

# as most things ML this is really about probability of something happening 
# based on previous knowledge, aka previous experperience. 
# good stuff https://en.wikipedia.org/wiki/Bayes%27_theorem 

# check working directory
import os
os.getcwd()

# =============================================================================
# # Importing the libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# # Importing the dataset
# =============================================================================
dataset = pd.read_csv('Social_Network_Ads.csv')
# lets look
'''
in: dataset
Output:
      User ID  Gender  Age  EstimatedSalary  Purchased
0    15624510    Male   19            19000          0
1    15810944    Male   35            20000          0
'''
# we'll build a model that will predict if a user will buy the SUV or not
# X based on Age and Estimated Salary are the matrix of features we are learning 
# y the dependent varialbe is purchase YES/NO 1/0
# in Python data starts with 0 zero
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# =============================================================================
# # Splitting the dataset into the Training set and Test set 25%
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# =============================================================================
# # Feature Scaling - yes as variables don't have the same units
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# # Fitting the classifier to the Training set - Naive Bayes this time
# =============================================================================
# Step 1 import necessary model and/or functions
from sklearn.naive_bayes import GaussianNB
# Step 2 create our object; 
# we'll use the Gaussian or RBF kernel, random state allows us
# all in class to get the same experience
classifier = GaussianNB() # note in ? NO Arguments
# Step 3 fit object to our data
classifier.fit(X_train, y_train)


# =============================================================================
# # Predicting the Test set results
# =============================================================================
# y is the dependent varialbe of purchase YES/NO 1/0
# here we use the predict function of the classifier object
y_pred = classifier.predict(X_test)
y_predtr = classifier.predict(X_train)

# =============================================================================
# # Making the Confusion Matrix
# =============================================================================
# Wikipedia https://en.wikipedia.org/wiki/Confusion_matrix
# basically a table of predictions vs actual or vice versa
# we'll import a function called confusion_matrix
from sklearn.metrics import confusion_matrix

# and now we'll compute comparing y_test and y_pred
cm = confusion_matrix(y_test, y_pred)
cm
'''
Have a look
cm
Out[14]: 
array([[65,  3],
       [ 7, 25]])
'''
# the colums are for 1 and 0 
# 90 correct predictions
# 10 incorrect predictions

# now we can run the CM on the Y Train data with the predictor from the X train
# this is where the model was trained :) 
cmtr = confusion_matrix(y_train, y_predtr)
cmtr
'''
cmtr
Out[16]: 
array([[174,  15],
       [ 20,  91]])
'''
# 35 out of 300 are incorrect, eh about the same as test

# =============================================================================
# # Visualising the Training set results
# =============================================================================
from matplotlib.colors import ListedColormap
# think of this bit as declaring X_set and y_set in our code so we can easily
# change the data being used by the code
X_set, y_set = X_train, y_train
# next we create the background/region
# the min and max -1 and +1 help to pull the data off the edges of the graph so 
# they aren't squeezed to the edges of the graph
# each pixel (step 0.01) on the graph we applied or classifier the red or green background
# think of the step as the resolution, the step between each 'point' on the graph
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# this is the MAGIC the countourf creates the line between red and green and we use the predict
# to work out if each point is class 0 (red) or class 1 (green)
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# this plots the limits of X and Y data points
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# next the loop goes through the real data points in X_set and y_set, 0 or 1's
# and we're plotting them on top of the background/region created above
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# =============================================================================
# # Visualising the Test set results
# =============================================================================
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

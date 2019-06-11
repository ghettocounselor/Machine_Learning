# Artificial Neural Network

# How do neural networks work?
# Lecture 219 https://www.udemy.com/machinelearning/learn/lecture/6760386 
# Outline of Neural Netword Activation Function use
from PIL import Image
img = Image.open("ANN_OutlineOfProcess.png")
img.show()

# How do Neural Networks learn? The Cost Function is the difference between
# the y^ prediction and y the actual value, so the lower the Cost Function the
# higer the accuracy of the NN. This process is refferred to as back propagation. 
# Lecture 220 https://www.udemy.com/machinelearning/learn/lecture/6760388
from PIL import Image
img = Image.open("ANN_HowDoTheyLearn.png")
img.show()
from PIL import Image
img = Image.open("ANN_HowDoTheyLearn2.png")
img.show()
from PIL import Image
img = Image.open("ANN_MultipleHiddenLayers.png")
img.show()

# nicely done post : https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications  

# Gradient Descent, Curse of Dimensionality, nuances to adjustment of weights. 
# requires a convex relationship of y and Cost Function. Gradient Descent is 
# also called batch gradient descent. 
# Lecture 221 https://www.udemy.com/machinelearning/learn/lecture/6760390

# Stochastic Gradient Descent, does not require convex relationship. This is NOT
# a batch process, it's running each row (each weight) at a time. This allows 
# the proces to find the BEST Cost Function even if relationship of y to C is not
# convex. 
# Lecture 222 https://www.udemy.com/machinelearning/learn/lecture/6760392

# Backpropagation 
# Lecture 223 https://www.udemy.com/machinelearning/learn/lecture/6760394
# good post http://neuralnetworksanddeeplearning.com/chap2.html#the_four_fundamental_equations_behind_backpropagation 

# check working directory
import os
os.getcwd()

# Steps
from PIL import Image
img = Image.open("ANN_Steps.png")
img.show()

# =============================================================================
# # Installing Keras
# =============================================================================
# Enter the following command in a terminal (or anaconda prompt for Windows users): 
# conda install -c conda-forge keras
# TensorFlow https://www.tensorflow.org/

# =============================================================================
# # Part 1 - Data Preprocessing
# =============================================================================
# Lecture 228 https://www.udemy.com/machinelearning/learn/lecture/6127182 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# # Importing the dataset
# =============================================================================
dataset = pd.read_csv('Churn_Modelling.csv')
# Let's have a look at the variables. 
from PIL import Image
img = Image.open("BankCustomerData.png")
img.show()
# we have 10 Independent variables from index 3 to 13 (in effect columns 3-12)
# the Dependent variable is 13 'Exited'
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# let's look
X
'''
Out[11]: 
array([[619, 'France', 'Female', ..., 1, 1, 101348.88],
       [608, 'Spain', 'Female', ..., 0, 1, 112542.58],
       [502, 'France', 'Female', ..., 1, 0, 113931.57],
       ...,
       [709, 'France', 'Female', ..., 0, 1, 42085.58],
       [772, 'Germany', 'Male', ..., 1, 0, 92888.52],
       [792, 'France', 'Female', ..., 1, 0, 38190.78]], dtype=object)
'''
y
'''
Out[12]: array([1, 0, 1, ..., 1, 1, 0])
'''
# =============================================================================
# # Encoding categorical data - Geography (country) & Gender
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Geography (country) & Gender need to be Encoded because they don't 'mean' anything
# as their name, we need 1 or 0's ;) 
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
X
'''
Out[15]: 
array([[619, 0, 0, ..., 1, 1, 101348.88],
       [608, 2, 0, ..., 0, 1, 112542.58],
       [502, 0, 0, ..., 1, 0, 113931.57],
       ...,
       [709, 0, 0, ..., 0, 1, 42085.58],
       [772, 1, 1, ..., 1, 0, 92888.52],
       [792, 0, 0, ..., 1, 0, 38190.78]], dtype=object)
'''
# =============================================================================
# # Build Dummy Variables for Country - again we need rows that answer yes or no
# =============================================================================
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# avoid dummy variable trap by removing one of the 3 columns created for Country
X = X[:, 1:]

# =============================================================================
# # Splitting the dataset into the Training set and Test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# Feature Scaling - needed in order to ease all the calculations we'll be doing
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# # Part 2 - Now let's make the ANN!
# =============================================================================

# =============================================================================
# # Importing the Keras libraries and packages
# =============================================================================
# Lecture 229 https://www.udemy.com/machinelearning/learn/lecture/6128646

import keras
from keras.models import Sequential
from keras.layers import Dense

# =============================================================================
# # Initialising the ANN as a sequence of layers (as opposed to a graph)
# =============================================================================
# Lecture 230 https://www.udemy.com/machinelearning/learn/lecture/6128792
# Step 1 create and object of the Sequential model. Since we are creating a 
# claisification NN we'll call it classifier
classifier = Sequential()

# Step 2 Adding the input layer and the first hidden layer
# Lecture 231 https://www.udemy.com/machinelearning/learn/lecture/6210322
# we describe the # of inputs = to our Independent variables (11) and we'll use
# the activation function Rectifier is relu
# units rule of thumb input varialbes (11) + outputs (1) / 2 because there are two groups so 6
# kernel initializer of uniform handles the setup of the weights
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# for help keras.layers.core.Dense

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
# Lecture 233 https://www.udemy.com/machinelearning/learn/lecture/6132050
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# Lecture 234 https://www.udemy.com/machinelearning/learn/lecture/6210328
# the optimizer is the stochastic decent algorithm, we'll use one called 'adam'
# loss is the loss function within the 'adam' algorithm. This loss function will be 
# optimized by the NN. Metrics is the criteria that will be used to improve the performance. 
# we're telling the NN to use the accuracy as the criteria for improvement. 
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# Lecture 235 https://www.udemy.com/machinelearning/learn/lecture/6210344
# epochs is the number of times we'll run through the process of updating the  
# weights
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
# Lecture 236 https://www.udemy.com/machinelearning/learn/lecture/6135916
y_pred = classifier.predict(X_test)
# because 'predict' brings back probabilities but what we need is a yes/no
# so we need to choose a threshold where something is true, we'll go with 0.5
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''
array([[1538,   57],
       [ 257,  148]])
'''
# accuracy
# (correct) / (Total observations)
(1538 + 148) / 2000
'''
0.843
'''




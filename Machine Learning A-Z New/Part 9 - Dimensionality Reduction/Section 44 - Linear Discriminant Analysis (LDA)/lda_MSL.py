# Linear Discriminant Analysis

# Lecture 272 Intuition https://www.udemy.com/machinelearning/learn/lecture/10628136
# Lecture 273 https://www.udemy.com/machinelearning/learn/lecture/6270778

from PIL import Image
img = Image.open("LDAvPCA.png")
img.show()

# Both Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA) are  
# linear transformation techniques that are commonly used for dimensionality reduction. 
# PCA can be described as an “unsupervised” algorithm, since it “ignores” class labels and 
# its goal is to find the directions (the so-called principal components) that maximize 
# the variance in a dataset. In contrast to PCA, LDA is “supervised” and computes the 
# directions (“linear discriminants”) that will represent the axes that that maximize the 
# separation between multiple classes. 
# https://sebastianraschka.com/Articles/2014_python_lda.html

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
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

from PIL import Image
img = Image.open("DatasetInformation.png")
img.show()

# =============================================================================
# # Splitting the dataset into the Training set and Test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# # Feature Scaling
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# # Applying LDA
# =============================================================================
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2) # we'll use 2 so we can visualize
# fit the LDA object to our Train and Test sets
X_train = lda.fit_transform(X_train, y_train) # NOTE we include the Independent variable
# this is why LDA is considered superised 
X_test = lda.transform(X_test)

# =============================================================================
# # Fitting Logistic Regression to the Training set
# =============================================================================
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) # random state so we all get same #'s
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
''' perfect!
array([[14,  0,  0],
       [ 0, 16,  0],
       [ 0,  0,  6]])
'''
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression - LDA (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression - LDA (Test set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()
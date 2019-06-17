# PCA

# Lecture 264 PCA Intuition
# https://www.udemy.com/machinelearning/learn/lecture/10628128 

# Great presentation and tutorial
# https://plot.ly/ipython-notebooks/principal-component-analysis/ 

# Another explanation of topic
# http://setosa.io/ev/principal-component-analysis/
# home page of this stuff http://setosa.io/ev/ 

# Example of another implementation
# https://sebastianraschka.com/Articles/2014_pca_step_by_step.html

# PCA Wikipedia
# https://en.wikipedia.org/wiki/Principal_component_analysis

# PCA in a nutshell; what we are doing is taking a large number of independent 
# variables and extracting them down to a core group of new independent variables
# that best describe the relationship (most variance) of the data in the dataset. 
# because this extraction is down without knowledge of the dependent variable 
# PCA is considered un-supervised. 

# Lecture 266 https://www.udemy.com/machinelearning/learn/lecture/6270772

# =============================================================================
# Importing the libraries
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
# # Applying PCA - our dimentionality reduction technique
# =============================================================================
# Lecture 267 https://www.udemy.com/machinelearning/learn/lecture/6270776
from sklearn.decomposition import PCA
# create an object of the PCA class, we'll call it pca
# we define the number of extracted features we would like, we'll choose 2
# so we can later visualize. BUT first we'll do some work. 
'''
pca = PCA(n_components = None)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
'''
from PIL import Image
img = Image.open("explained_variance.png")
img.show()

# change to 2 based on analysis. we'll extract to 2 variables
pca = PCA(n_components = 2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

# =============================================================================
# # Fitting Logistic Regression to the Training set
# =============================================================================
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# =============================================================================
# # Predicting the Test set results
# =============================================================================
y_pred = classifier.predict(X_test)

# =============================================================================
# # Making the Confusion Matrix
# =============================================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# two column cm explained
from PIL import Image
img = Image.open("Confusion_Matrix_Explained.png")
img.show()
# we have 3 possible outcomes with the 3 segments of customers
# each column represents a customer segment and the rows represent which 
# segment was chosen by the algorithm when the segment was the one represented
# by the column. 
array([[14,  0,  0],
       [ 1, 15,  0],
       [ 0,  0,  6]])
# for column 1, customer segment 1, we have 14 correct, 1 time customer seg
# 2 was selected but it was 1. column 2, customer segment 2 was perfect
# as was 3
# 35 / 36 is a pretty good accuracy. 

# =============================================================================
# # Visualising the Training set results
# =============================================================================
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             # here we need three colors one for each segment (prediction regions)
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
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
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# =============================================================================
# # Predicting the Train set results
# =============================================================================
y_predTr = classifier.predict(X_train)

# =============================================================================
# # Making the Confusion Matrix Train
# =============================================================================
from sklearn.metrics import confusion_matrix
cmTr = confusion_matrix(y_train, y_predTr)
'''
array([[43,  2,  0],
       [ 2, 52,  1],
       [ 0,  0, 42]])
'''
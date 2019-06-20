# Kernel PCA

# Lecture 277 https://www.udemy.com/machinelearning/learn/lecture/6270782
# Remember that we have non-linearly separable data here so we don't want
# to use LDA or PCA (alone) 

# Let's remember our Logistic Regression Model example from classification 
# part of the course. 
from PIL import Image
img = Image.open("LogisticRegression_dataExplained.png")
img.show()

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
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# =============================================================================
# # Splitting the dataset into the Training set and Test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# =============================================================================
# # Feature Scaling
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =============================================================================
# # Applying Kernel PCA - important step here
# =============================================================================
# this will create a space where our data will be linearly separable. 
from sklearn.decomposition import KernelPCA
# components of 2 to allow for visability, the rbf is Gausian kernal
kpca = KernelPCA(n_components = 2, kernel = 'rbf') 
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

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
cm
#
# Out[3]: 
# array([[64,  4],
#       [ 6, 26]])
# ```
# =============================================================================
# # Visualising the Training set results
# =============================================================================
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression with Kernal PCA (Training set)')
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
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression with Kernal PCA (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# =============================================================================
# Logistic regression Before and After Kernel PCA
# =============================================================================
from PIL import Image
img = Image.open("LogReg_BeforeAfterKernelPCA.png")
img.show()
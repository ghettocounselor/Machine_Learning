# K-Means Clustering

# K Means Intuition Lecture 137 https://www.udemy.com/machinelearning/learn/lecture/5714416

# https://en.wikipedia.org/wiki/K-means_clustering

# K Means Random Initialization Trap Lecture 138 https://www.udemy.com/machinelearning/learn/lecture/5714420

# K Means Selecting the number of clusters Lecture 139
# within cluster sum of squares

# K Means Clustering Python Lecture 141 https://www.udemy.com/machinelearning/learn/lecture/5685588

'''
Clustering is similar to classification, but the basis is different. In Clustering you don’t know what you are looking for, and you are trying to identify some segments or clusters in your data. When you use clustering algorithms on your dataset, unexpected things can suddenly pop up like structures, clusters and groupings you would have never thought of otherwise.
'''

# highlight the % through -f to reset
# %reset -f

# also in R on RPubs http://rpubs.com/markloessi
# check working directory
import os
os.getcwd()

# =============================================================================
# # Importing the libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')

# view dataset from variable explorer, screen grab and save image then display
from PIL import Image
img = Image.open("/Users/markloessi/Machine_Learning/Machine Learning A-Z New/Part 4 - Clustering/MallCustomer_Task.png")
img.show()

# now we want to build an array of our two columns we want to test  
X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values

# =============================================================================
# # Splitting the dataset into the Training set and Test set 25%
# =============================================================================
# not doing this for KMeans
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# =============================================================================
# # Feature Scaling - yes as variables don't have the same units
# =============================================================================
# not doing this for KMeans
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# =============================================================================
# Using the elbow method to find the optimal number of clusters - 2 dimensions
# =============================================================================

# What' the Elbow Method
from PIL import Image
img = Image.open("/Users/markloessi/Machine_Learning/TheElbowMethodClusteringNums.png")
img.show()

from sklearn.cluster import KMeans
# calculate and store in an array the within clusters sum of squares
wcss = []
# loop through our clusters (we use 11 because it will be excluded and we are using 10 clusters)
for i in range(1, 11):
    # creating object of our KMeans class
    # init = the initialization method we'll use, we'll use k-means++ 
    # as noted in our intuition section to avoid the initialization trap
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                    max_iter = 300, n_init = 10, random_state = 0)
    # fit object to our data
    kmeans.fit(X)
    # append the WCSS to our list
    # inertia is the attribute that is within kmeans that will calculate wcss
    wcss.append(kmeans.inertia_)
# plot the elbow method graph, x axis is range and y is wcss
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# =============================================================================
# Fitting K-Means to the dataset with new clusters
# =============================================================================
# review from above makes it look like 5 is the right number
# make a new object with new clusters. 
kmeans = KMeans(n_clusters = 5, init = 'k-means++',
                    max_iter = 300, n_init = 10, random_state = 0)
# fit predict will work out which cluster each customer belongs too
y_kmeans = kmeans.fit_predict(X)

# =============================================================================
# Visualising the clusters for each customer
# =============================================================================
'''
syntax; basically separating each cluster out of y_kmeans where first number
in the pair 0, 0 is the number and the second is the column and sindce we have 1 
column that will always be 0 here. Other variables are sensible, size is only odd 
item in the listing. 
'''
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# =============================================================================
# Visualising the clusters - making more readable labels
# =============================================================================
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
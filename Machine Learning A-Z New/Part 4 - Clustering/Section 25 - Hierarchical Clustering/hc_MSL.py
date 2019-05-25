# Hierarchical Clustering

# HC Intuition Lecture 143  https://www.udemy.com/machinelearning/learn/lecture/5714428

# also in R on RPubs http://rpubs.com/markloessi

# check working directory
import os
os.getcwd()

'''
In data mining and statistics, hierarchical clustering (also called hierarchical cluster analysis or HCA) is a method of cluster analysis which seeks to build a hierarchy of clusters. Strategies for hierarchical clustering generally fall into two types:[1]

Agglomerative: This is a "bottom-up" approach: each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.
Divisive: This is a "top-down" approach: all observations start in one cluster, and splits are performed recursively as one moves down the hierarchy.

In general, the merges and splits are determined in a greedy manner. The results of hierarchical clustering are usually presented in a dendrogram.

https://en.wikipedia.org/wiki/Hierarchical_clustering
'''

# Hierachical Clustering Steps
from PIL import Image
img = Image.open("hc_steps.png")
img.show()

# What's going on here
Lets remember Euclidean Distances
from PIL import Image
img = Image.open("Euclidean_Distances.png")
img.show()

# see minute 6:36 in intuition lecture above for visual walk through of steps
from PIL import Image
img = Image.open("hc_steps.png")
img.show()

# Working out the distances between clusters
from PIL import Image
img = Image.open("Euclidean_Distances_Options.png")
img.show()

'''
What the Hierachical Clustering algorithm does while it walks through the steps
above is stores the memory of the steps in a Dendrogram :-)

https://en.wikipedia.org/wiki/Dendrogram

How the Dendogram works https://www.udemy.com/machinelearning/learn/lecture/5714432
Basically how is the Dendogram created. 
'''

from PIL import Image
img = Image.open("How_Dendogram_Forms.png")
img.show()

'''
What we do is set a dissimilarity threshold based on a certain Euclidean Distance. 
We can work how where to set this by visualizing the Dendogram. 
'''

# Dendogram Threshold
from PIL import Image
img = Image.open("Dendogram_Threshold.png")
img.show()

# Threshold into Cluster Count
from PIL import Image
img = Image.open("Threshold_intoClusterCount.png")
img.show()

# =============================================================================
# # Importing the libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Importing the dataset
# =============================================================================
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
# not doing this here
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
# Using the dendrogram to find the optimal number of clusters
# =============================================================================

# What's the dendrogram
from PIL import Image
img = Image.open("How_Dendogram_Forms.png")
img.show()

# we'll use scipy to use the dendogram tools 
import scipy.cluster.hierarchy as sch
# The Ward method attempts to minimize the variance within each cluster
# this is like the Withing Cluster Sum of Squares
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# interpret our Threshold
from PIL import Image
img = Image.open("Dendogram_OptimalClusters.png")
img.show()

# =============================================================================
# Fitting Hierarchical Clustering to the dataset
# =============================================================================
# we worked out 5 clusters
from sklearn.cluster import AgglomerativeClustering
# create our object 
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
# fit to our data
y_hc = hc.fit_predict(X)

# =============================================================================
# Visualising the clusters - will only work with 2 dimensions
# =============================================================================
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# =============================================================================
# Visualising the clusters - making more readable labels
# =============================================================================
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Clustering Pros Cons
from PIL import Image
img = Image.open("Clustering_ProsCons.png")
img.show()

# to clean up - highlight the % through -f to reset
# %reset -f and then run cntrl L in console
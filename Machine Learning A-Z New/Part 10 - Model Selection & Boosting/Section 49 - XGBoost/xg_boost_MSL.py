# XGBoost with k-Fold Cross Validation

# =============================================================================
# # Benefits of XGBoost
# =============================================================================
# High performance
# Fast exectution 
# We can keep all the interpretation of your problem, aka there's isn't all the 
# fixing of stuff like country by making dummy variables. 

# Installing XGBoost
# Enter the following command in a terminal (or anaconda prompt for Windows users): 
# conda install -c conda-forge xgboost

# =============================================================================
# # Importing the libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# # Importing the dataset - Bank churn business case
# =============================================================================
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# =============================================================================
# # Encoding categorical data
# =============================================================================
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# =============================================================================
# # Splitting the dataset into the Training set and Test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# # No feature scalling necessary
# =============================================================================

# =============================================================================
# # Fitting XGBoost to the Training set
# =============================================================================
from xgboost import XGBClassifier
classifier = XGBClassifier()
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
'''
array([[1521,   74],
       [ 197,  208]])
'''

# =============================================================================
# # Applying k-Fold Cross Validation
# =============================================================================
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
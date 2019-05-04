#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 06:38:26 2019

@author: markloessi
"""
# from lecture 48 https://www.udemy.com/machinelearning/learn/lecture/9201998

# =============================================================================
# code from lectures for below backwards elim
# =============================================================================
import numpy as np
import pandas as pd

dataset = pd.read_csv('/Users/markloessi/Machine_Learning/Machine Learning A-Z New/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS .summary()
# push results out to CSV for storing
f = open('X_opt_SummaryAutoPath.csv','w')
f.write(regressor_OLS .summary().as_csv())
f.close()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# =============================================================================
# Backward Elimination with p-values only
# =============================================================================
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05 # adjust the desired P-value here
X_optP = X[:, [0, 1, 2, 3, 4, 5]]
X_ModeledP = backwardElimination(X_optP, SL)
# extra code from a question in lecture 48
def powerPredictor(x):
    x = backwardElimination(X,SL)
    p = []
    keys = dataset.keys()
    for i in range(0,len(x[0])):
        for j in range(0,len(keys)-1):
            b = np.equal(x[:,i],dataset[keys[j]].values)[0] #np.equal() will return an array of booleans
            if b and True:
                p.append(keys[j])
    return p

features_usedP = powerPredictor(X_optP)
features_usedP
# =============================================================================
# Backward Elimination with p-values and Adjusted R Squared
# =============================================================================

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_optPRS = X[:, [0, 1, 2, 3, 4, 5]]
X_ModeledPRS = backwardElimination(X_optPRS, SL)
# extra code from a question in lecture 48
def powerPredictor(x):
    x = backwardElimination(X,SL)
    p = []
    keys = dataset.keys()
    for i in range(0,len(x[0])):
        for j in range(0,len(keys)-1):
            b = np.equal(x[:,i],dataset[keys[j]].values)[0] #np.equal() will return an array of booleans
            if b and True:
                p.append(keys[j])
    return p

features_usedPRS = powerPredictor(X_optPRS)
features_usedPRS
# Apriori

# Intuition Lecture 159 https://www.udemy.com/machinelearning/learn/lecture/6455322
# Python Lectures
# 164 step 1 https://www.udemy.com/machinelearning/learn/lecture/5966530
# 165 step 2 https://www.udemy.com/machinelearning/learn/lecture/5969220
# 166 step 3 https://www.udemy.com/machinelearning/learn/lecture/5971018

https://en.wikipedia.org/wiki/Association_rule_learning

https://en.wikipedia.org/wiki/Apriori_algorithm

'''
# Overview
The Apriori (reference to prior knowledge) algorithm was proposed by Agrawal and Srikant in 1994. Apriori is designed to operate on databases containing transactions, for example, collections of items bought by customers, or details of a website frequentation or IP addresses. 

Other algorithms are designed for finding association rules in data having no 
transactions (Winepi and Minepi), or having no timestamps (DNA sequencing). 
Each transaction is seen as a set of items (an itemset). Given a threshold 
{\displaystyle C} C, the Apriori algorithm identifies the item sets which are 
subsets of at least {\displaystyle C} C transactions in the database.

Apriori uses a "bottom up" approach, where frequent subsets are extended one 
item at a time (a step known as candidate generation), and groups of candidates 
are tested against the data; people who bought beer also bought diapers, people who bought diapers also bought laundry detergent, and so on. The algorithm terminates when no further successful extensions are found.

Apriori uses breadth-first search and a Hash tree structure to count candidate 
item sets efficiently. It generates candidate item sets of length {\displaystyle k} 
k from item sets of length {\displaystyle k-1} k-1. Then it prunes the candidates 
which have an infrequent sub pattern. According to the downward closure lemma, 
the candidate set contains all frequent {\displaystyle k} k-length item sets. 
After that, it scans the transaction database to determine frequent item sets 
among the candidates.
'''

# also in R on RPubs http://rpubs.com/markloessi

# check working directory
import os
os.getcwd()

# =============================================================================
# Importing the libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Import dataset 
# =============================================================================
dataset = pd.read_csv('Market_Basket_Optimisation.csv')
# lets look
dataset
'''
Out[6]: 
                 shrimp            almonds  ... spinach olive oil
0               burgers          meatballs  ...     NaN       NaN
1               chutney                NaN  ...     NaN       NaN
2                turkey            avocado  ...     NaN       NaN
'''
# note no header row is present, so lets fix that
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
dataset
'''
Out[8]: 
                     0                  1   ...       18         19
0                shrimp            almonds  ...  spinach  olive oil
1               burgers          meatballs  ...      NaN        NaN
2               chutney                NaN  ...      NaN        NaN
3                turkey            avocado  ...      NaN        NaN
'''
'''
What we are looking at are the individual transactions for the store in a week.
What we would like to work out is based on these transactions, what things can
we intuit that people will buy if they buy another thing first. 
'''
# =============================================================================
# Data Preprocessing
# =============================================================================
'''
also note all the nan or null items, we need to import the dataset in a particular 
way. We have an array, but what apriori is expecting is a list of lists. So, we
need to remove all the nan's from each row and create a list of lists of all the
items for each transaction. 
'''
# we'll call our list transactions and start off with an empty list
transactions = []
# now we loop through the transactions array starting at 0 up to 7501
for i in range(0, 7501):
    # next we append our string created by defining j by describing values
    # as values[i,j] where j 'is' the range (0,20) of i ;-) 
    # in english, take each transaction 'i' and grab all the items 'j' and
    # stick them into the transactions list
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    # note this is a loop in a loop
    
    #if we want to clean off the nan from the listing we can do this
transactionsNoNaN = [ [item for item in row if not pd.isnull(item) ] for i,row in dataset.iterrows()]

# =============================================================================
# apriori will not need splitting 
# =============================================================================
# =============================================================================
# apriori will not need feature scaling
# =============================================================================

# The steps to work through to tune the algorithm. 
from PIL import Image
img = Image.open("Apriori_steps.png")
img.show()

# =============================================================================
# Training Apriori on the dataset
# =============================================================================
# note APYORI file in our folder is necessary to create the rules
from apyori import apriori
# have a look
?apriori # see below for details
# we'll make an object called rules to hold our rules from the apriori 
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

'''
The setup of the rule variables is dependent on your buisness decision(s) you are
trying to make. Seems there is a bit of playing around here. 
'''
# min_support - this is how many times per day in our case. Let's look at 3
3*7/7501
'''
Out[20]: 0.0027996267164378083
'''
# 4 times a day, this is the 'support' for an item purchased X times a day
4*7/7501
'''
Out[21]: 0.0037328356219170776
'''
'''
min_confidence this literally is the likelihood, think of it as a percent
example 0.8 means that 80% of the time the 'rule' will be true. The higher
the confidence the more obvious the rules will be. We don't want obvious
we want insights, so there's a balance. 

The combination of support and confidence is necessary. 
'''
'''
Lift : https://en.wikipedia.org/wiki/Association_rule_learning#Lift
'''
# =============================================================================
# Visualising the results
# =============================================================================
# in R we needed to sort, but here the list is already sorted by its own 
# relevance criterion
results = list(rules)
# to view in console
results
# that's messy, lets clean up a bit by making a listing of the variables in a list
results_list = []
for i in range(0, len(results)):       
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t'
                        + str(results[i][1])+ '\nLift:\t' + str(results[i][2]))

# The steps to work through to tune the algorithm. 
from PIL import Image
img = Image.open("Apriori_Rules_Python.png")
img.show()


# Reading the rules 
from PIL import Image
img = Image.open("Apriori_ReadResults.png")
img.show()

# =============================================================================
# info on apriori package in file provided with course materials
# =============================================================================
'''
Signature: apriori(transactions, **kwargs)
Docstring:
Executes Apriori algorithm and returns a RelationRecord generator.

Arguments:
    transactions -- A transaction iterable object
                    (eg. [['A', 'B'], ['B', 'C']]).

Keyword arguments:
    min_support -- The minimum support of relations (float). 
    min_confidence -- The minimum confidence of relations (float).
    min_lift -- The minimum lift of relations (float).
    min_length -- The minimum length of the relation (integer). Basically the min length of our list. 
    max_length -- The maximum length of the relation (integer).
File:      ~/Machine_Learning/Machine Learning A-Z New/Part 5 - Association Rule Learning/Section 28 - Apriori/Apriori_Python/apyori.py
Type:      function
'''












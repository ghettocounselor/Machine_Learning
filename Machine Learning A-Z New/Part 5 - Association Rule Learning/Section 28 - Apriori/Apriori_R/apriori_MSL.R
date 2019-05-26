# Apriori

# Intuition Lecture 159 https://www.udemy.com/machinelearning/learn/lecture/6455322
# https://en.wikipedia.org/wiki/Association_rule_learning
# https://en.wikipedia.org/wiki/Apriori_algorithm

'''Overview
The Apriori algorithm was proposed by Agrawal and Srikant in 1994. Apriori is 
designed to operate on databases containing transactions, for example, collections 
of items bought by customers, or details of a website frequentation or IP addresses. 

Other algorithms are designed for finding association rules in data having no 
transactions (Winepi and Minepi), or having no timestamps (DNA sequencing). 
Each transaction is seen as a set of items (an itemset). Given a threshold 
{\displaystyle C} C, the Apriori algorithm identifies the item sets which are 
subsets of at least {\displaystyle C} C transactions in the database.

Apriori uses a "bottom up" approach, where frequent subsets are extended one 
item at a time (a step known as candidate generation), and groups of candidates 
are tested against the data. The algorithm terminates when no further successful 
extensions are found.

Apriori uses breadth-first search and a Hash tree structure to count candidate 
item sets efficiently. It generates candidate item sets of length {\displaystyle k} 
k from item sets of length {\displaystyle k-1} k-1. Then it prunes the candidates 
which have an infrequent sub pattern. According to the downward closure lemma, 
the candidate set contains all frequent {\displaystyle k} k-length item sets. 
After that, it scans the transaction database to determine frequent item sets 
among the candidates.
'''
# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
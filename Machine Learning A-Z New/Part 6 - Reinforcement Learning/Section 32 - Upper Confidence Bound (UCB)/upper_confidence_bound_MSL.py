# Upper Confidence Bound - example of Reinforement Learning

# Intuition Lecture 172 https://www.udemy.com/machinelearning/learn/lecture/6456832

# Python Lecture 174 https://www.udemy.com/machinelearning/learn/lecture/5984442
# lecture 175 https://www.udemy.com/machinelearning/learn/lecture/6017492

# Thompson Sampling wikipedia
# https://papers.nips.cc/paper/4909-eluder-dimension-and-the-sample-complexity-of-optimistic-exploration.pdf

# also in R on RPubs http://rpubs.com/markloessi
# Good notes and questions https://github.com/ghettocounselor/Machine_Learning/blob/master/Machine-Learning-A-Z-Q-A.pdf 

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
# This is a similation dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# interpret the dataset 
from PIL import Image
img = Image.open("Dataset_interpretation.png")
img.show()

# =============================================================================
# # Implementing RANDOM Selection algorithm
# =============================================================================
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward = total_reward + reward
# lets take a look at how random did
total_reward
'''
total_reward
Out[9]: 1274
'''
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
plt.savefig('RandomOutput.png')

# =============================================================================
# Implementing UCB
# =============================================================================
# interpret the dataset 
from PIL import Image
img = Image.open("UCB_Algorithm_Slide.png")
img.show()
# lecture 175 https://www.udemy.com/machinelearning/learn/lecture/6017492
# lecture 176 https://www.udemy.com/machinelearning/learn/lecture/6021982
import math
N = 10000   # number of rounds we want
d = 10      # number of arms (ads)
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
# start our first loop
for n in range(0, N): # N is the total number of rounds, so we start at zero and go to N
    # run through each version of the ad
    ad = 0 # initialize the variable 'ad' which will count the wrapping through the for loop
    max_upper_bound = 0 # part of step 3 in slide
    for i in range(0, d):
        # from the algorithm slide we make our computations
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                        # because in python the 1st round will be zero so we add 1
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            # upper confidence bound
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 # we're going to set the upper bound very large at first
            # this will for the algorithm to never force 
        # in each round we'll check the upper bound against the previous max
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    # step 3 in slide starts here
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
total_reward
'''
total_reward - much better!
Out[12]: 2178
'''

# lecture 177 https://www.udemy.com/machinelearning/learn/lecture/5997012
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
plt.savefig('UCBOutput.png')
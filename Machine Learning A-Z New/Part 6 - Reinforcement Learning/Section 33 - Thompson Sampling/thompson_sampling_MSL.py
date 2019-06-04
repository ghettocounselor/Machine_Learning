# Thompson Sampling

# Intuition of Reinforcement Learning Lecture 172 https://www.udemy.com/machinelearning/learn/lecture/6456832

# Thompson Sampling Intuition Lecture 182 https://www.udemy.com/machinelearning/learn/lecture/6456840

# UCB vs Thompson Sampling Lecture 183 https://www.udemy.com/machinelearning/learn/lecture/6468288

# Thompson Sampling wikipedia
# https://papers.nips.cc/paper/4909-eluder-dimension-and-the-sample-complexity-of-optimistic-exploration.pdf

# also in R on RPubs http://rpubs.com/markloessi
# Good notes and questions https://github.com/ghettocounselor/Machine_Learning/blob/master/Machine-Learning-A-Z-Q-A.pdf 

# =============================================================================
# Lecture links for this file
# =============================================================================
# Lecture 185 https://www.udemy.com/machinelearning/learn/lecture/6027254
# Lecture 186 https://www.udemy.com/machinelearning/learn/lecture/6012490

# check working directory
import os
os.getcwd()

# =============================================================================
# Importing the libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# =============================================================================
# Random processing
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
total_reward
'''
Out[24]: 1257
'''

# Visualising the random results
plt.hist(ads_selected)
plt.title('Histogram of ads selections (random)')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
plt.savefig('RandomOutput.png')

# =============================================================================
# steps for Thompson Sampling
# =============================================================================
from PIL import Image
img = Image.open("Thompson_Sampling_Slide.png")
img.show()
# =============================================================================
# # Implementing Thompson Sampling
# =============================================================================
# Lecture 185 https://www.udemy.com/machinelearning/learn/lecture/6027254
# Lecture 186 https://www.udemy.com/machinelearning/learn/lecture/6012490
import random
N = 10000
d = 10
ads_selected = []
# these bits here change from UCB to Thompson Sampling
numbers_of_rewards_1 = [0] * d # number of positive rewards
numbers_of_rewards_0 = [0] * d # number of negative rewards
# =========
total_reward = 0
for n in range(0, N):
    ad = 0
    # changes from UCB to Thompson Sampling
    max_random = 0
    for i in range(0, d):
        # this bit is the meat! we are calculating if the new data is better or worse than the original assumption of the distribution (see intuition) and if the relatity is better or worse we adjust. 
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
total_reward
'''
total_reward
Out[23]: 2607
'''
# =============================================================================
# # Visualising the results - Histogram
# =============================================================================
plt.hist(ads_selected)
plt.title('Histogram of ads selections (Thompson Sampling)')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
plt.savefig('ThompsonSamplingOutput.png')
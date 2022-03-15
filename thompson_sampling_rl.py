# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:30:43 2021

@author: BoosamG
"""

#Thompson Sampling Algorithm - RL
#Import Libraries

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

#Import Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson sampling
import random
N = 10000
d = 10
ads_selected = []
no_of_rewards_1 = [0] * d
no_of_rewards_0 = [0] * d
total_rewards = 0

for n in range(0,N):
    ad = 0
    max_random = 0
    for i in range(0,d):
        random_beta = random.betavariate(no_of_rewards_1[i]+1, no_of_rewards_0[i]+1)
        if (random_beta > max_random):
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward == 1:
        no_of_rewards_1[ad] = no_of_rewards_1[ad] +1
    else:
        no_of_rewards_0[ad] = no_of_rewards_0[ad] +1
    total_rewards = total_rewards+ reward


# Visualising the results
pt.hist(ads_selected)
pt.title('Histogram of ads selections')
pt.xlabel('Ads')
pt.ylabel('Number of times each ad was selected')
pt.show()
        
        
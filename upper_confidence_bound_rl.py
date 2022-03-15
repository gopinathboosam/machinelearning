# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:09:58 2021

@author: BoosamG
"""

#Upper Confidence Bound Algorithm - RL
#Import Libraries

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd

#Import Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


#Implementing UCB
import math as mt
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

for n in range(0,N):
    ad = 0
    max_upper_bound =0
    
    for i in range(0,d):
        if (numbers_of_selections[i]>0):
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = mt.sqrt(3/2 * mt.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound>max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward


# Visualising the results
pt.hist(ads_selected)
pt.title('Histogram of ads selections')
pt.xlabel('Ads')
pt.ylabel('Number of times each ad was selected')
pt.show()
            
            



    
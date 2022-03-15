# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:00:01 2020

@author: BoosamG
"""

#Eclat - ARL
#Import Libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

#Import Dataset
#Use Header function to include the 0th row in the excel sheet
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
txns =  []#making the dataset into list of lists
for i in range(0,7501):
    txns.append([str(dataset.values[i,j]) for j in range(0,20)])

#Train the model on apriori
from apyori import apriori
#min_support = take sample of 3 products in a day * 7 DAYS(a week) and divide by total products 7500
#min_confidence = thumb rule of 0.2
#min_lift  = thumb rule 3
#min and max lenght deals with number of products - if buy2 and get 1, then min and max is 3. if buy1, get1 and buy 2 get 2 involved, min is 2 and max is 4
rules = apriori(transactions = txns, min_support = 0.003 , min_confidence = 0.2 , min_lift = 3, min_length = 2, max_length = 2)

#Plot the rules
#Determine Results
results = list(rules)
results

#Plot the dataset
## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support'])

## Displaying the results non sorted
resultsinDataFrame

## Displaying the results sorted by descending lifts
resultsinDataFrame.nlargest(n = 10, columns = 'Support')
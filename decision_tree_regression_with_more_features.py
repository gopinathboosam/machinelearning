# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 07:47:02 2020

@author: BoosamG
"""

#Decision Tree Regression
#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

#Import Dataset
dataset = pd.read_csv('50_Startups_1.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Split the dataset into test and train set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.05, random_state = 1)
print(x_train)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(x_test)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(y_train)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(y_test)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
#Train dataset with DTR
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 0)
dtr.fit(x_train,y_train)

#Predict the results
y_pred = dtr.predict(x_test)
print(y_pred)

y_pred = dtr.predict([[130000,150000,333333]])
print(y_pred)

#Not significant for single feature dataset


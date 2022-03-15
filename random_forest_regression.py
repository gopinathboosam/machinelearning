# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 07:47:02 2020

@author: BoosamG
"""

#Random Forest Regression
#Decide a random K data in dataset - One Tree
#Train the model with K data
#Find the Y predict value from theabove trained model
#repeat the above steps. 
#Get the Average y value of all the repeated steps
#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

#Import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Train dataset with Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
#Decide the nuymber of Trees with random k Data. Deciding it as 10
rfr = RandomForestRegressor(n_estimators = 10, random_state=0 )
rfr.fit(x,y)

#Predict the results
y_pred = rfr.predict([[6.5]])
print(y_pred)

#Not significant for single feature dataset

#plot the graph with high resolution
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
pt.scatter(x,y,color='red')
pt.plot(x_grid, rfr.predict(x_grid), color ='blue')
pt.title("Random  Forest Model")
pt.xlabel('Level')
pt.ylabel('Salaries')
pt.show()
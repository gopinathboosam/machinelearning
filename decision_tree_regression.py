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
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Train dataset with DTR
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state = 0)
dtr.fit(x,y)

#Predict the results
y_pred = dtr.predict([[6.5]])
print(y_pred)

#Not significant for single feature dataset

#plot the graph with high resolution
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
pt.scatter(x,y,color='red')
pt.plot(x_grid, dtr.predict(x_grid), color ='blue')
pt.title("DTR Model")
pt.xlabel('Level')
pt.ylabel('Salaries')
pt.show()
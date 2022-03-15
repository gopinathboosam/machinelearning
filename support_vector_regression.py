# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 08:08:25 2020

@author: BoosamG
"""

#SVR - Support Vector Regression
#Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

#Import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
print(x)
print('*******************')
print(y)
print('*******************')

#Feature Scaling
#SVR doesn't have any coefficients that take care of the 
#sacling like Linear Regression Models. Hence, for SVR, we should do 
#Feature Scaling

#Reshape need to be done on dependent variable since StandardScaler accepts 2D array as Input params
y = y.reshape(len(y),1)
print(y)
print('*******************')
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

print(x)
print('*******************')
print(y)
print('*******************')


#Train the dataset with SVR
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(x,y)

#Predict the results 
#Since thevalues are in scale, we need to transform and inverse transform on the predictions
y_pred = sc_y.inverse_transform(svr.predict(sc_x.transform([[6.5]])))
print(y_pred)



#Visualize SVR
pt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red')
pt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(svr.predict(x)),color = 'blue')
pt.title('Salary vs Level')
pt.xlabel('Level')
pt.ylabel('Salary')
pt.show()

#Visualize with Smoother Curve
x_grid = np.arange(min(sc_x.inverse_transform(x)),max(sc_x.inverse_transform(x)),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
pt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red')
pt.plot(x_grid,sc_y.inverse_transform(svr.predict(sc_x.transform(x_grid))),color = 'blue')
pt.title('Salary vs Level')
pt.xlabel('Level')
pt.ylabel('Salary')
pt.show()
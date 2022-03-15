# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 08:02:08 2020

@author: BoosamG
"""

#SimpleRegression
#Import Libraries
import numpy as nm
import pandas as pd
import matplotlib.pyplot as pt

#Import Dataset
dataset = pd.read_csv('Salary_Data.csv')
#Decide Dependant and independant variables
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)

#No Missing variables - So no need to Deal with it
#No Categorical data - No need to Encode it

#Split Dataset into Train and Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0)
print('**********************************************************')
print(x_train)
print('**********************************************************')
print(x_test)
print('**********************************************************')
print(y_train)
print('**********************************************************')
print(y_test)
print('**********************************************************')

#FeatureSacling is not required on Simple Linear Regression since it has only one dependant variable

#Training SLR with Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


#Predict the Test Result
#Used incase if we need to predict the results for the unknown independant variable

y_pred = regressor.predict(x_test)
print(y_pred)


#Plot Training Set
pt.scatter(x_train, y_train, color = 'red', edgecolors='green')
pt.plot(x_train, regressor.predict(x_train), color = 'blue')
pt.title('Salary Vs Experience - [Training Set]')
pt.xlabel('Years of Experience')
pt.ylabel('Salary')
pt.show()

#Plot Test Set
pt.scatter(x_test, y_test, color = 'green')
pt.plot(x_train, regressor.predict(x_train), color = 'blue')
pt.title('Salary Vs Experience - [Test Set]')
pt.xlabel('Years of Experience')
pt.ylabel('Salary')
pt.show()

#Predict the Salary for the new data
x_new = ([0.5],[12],[20])
y_new = regressor.predict(x_new)
print(y_new)

#Getting the coefficent and constant values
print(regressor.coef_)
print(regressor.intercept_)
#Salary=regressor.coef_×YearsExperience+regressor.intercept_
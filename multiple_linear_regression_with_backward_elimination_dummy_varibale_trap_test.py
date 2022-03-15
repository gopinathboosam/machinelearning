# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 16:11:48 2020

@author: BoosamG
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 12:42:57 2020

@author: BoosamG
"""

#Multiple Linear Regression
#Import Linraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

#Import Dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#No Missing data

#Encode Categorical Data
#Encode Independant variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encode',OneHotEncoder(),[3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

#Dummy Variable Trap 
x = x[:,1:]
print(x)

#Split the dataset into train set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1, random_state = 1)

print('**********************************************************')
print(x_train)
print('**********************************************************')
print(x_test)
print('**********************************************************')
print(y_train)
print('**********************************************************')
print(y_test)
print('**********************************************************')

#Training the Multiple Linear Regression with train set
#Dummy Variable Trap - Dont worry
#Backward elimination technique - Dont worry
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the result
y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis = 1))#Concatenating vertically
print(np.concatenate((y_pred,y_test),axis = 0))#Horizantal Concatenation 

print(regressor.coef_)
print(regressor.intercept_)


#Backward Elimination Impl
#Select Significance level as 0.05. 
#Fit the model 
#Get the p values - Check whetehr P>SL, then remove the Predictor. Continue the Step 2
x = np.append(arr = np.ones((50,1)).astype(int), values = x , axis = 1)#Appending the xo as 1 to the columns as 1st column

import statsmodels.formula.api as sm
x_opt = x[:,[0,1,2,3,4,5]]#Optimal X - Which gives you the exact variables that needed
print(x)
x_opt = x_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()


x_opt = x[:,[0,1,3,4,5]]
print(x)
x_opt = x_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3,4,5]]
print(x)
x_opt = x_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3,5]]
print(x)
x_opt = x_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

x_opt = x[:,[0,3]]
print(x)
x_opt = x_opt.astype(np.float64)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()


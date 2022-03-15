# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:17:14 2020

@author: BoosamG
"""

#Polynomial Regression

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

#Import Dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:,1:2]
y = dataset.iloc[:,-1]

#Train the dataset with Linear Regression Model
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(x,y)

#Train the dataset with Polynomial Regression
#Use the POlynomial Parameters to define the degree of the order
from sklearn.preprocessing import PolynomialFeatures
pol_features = PolynomialFeatures(degree = 4)
x_pol = pol_features.fit_transform(x)
#we have created the degree for polynomial, hence use linear to make it polynomial linear
lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(x_pol,y)

#Visualize Linear Regression

pt.scatter(x,y,color='red')
pt.plot(x,lin_regressor.predict(x),color='blue')
pt.title('Linear Regression REsults')
pt.xlabel('Level')
pt.ylabel('Salary')
pt.show()

#Visualise Polynomial Regression
pt.scatter(x,y,color='red')
pt.plot(x,lin_regressor_2.predict(pol_features.fit_transform(x)),color='blue')
pt.title('Polynomial Regression REsults')
pt.xlabel('Level')
pt.ylabel('Salary')
pt.show()


#predict linear regression data
y_pred_lin = lin_regressor.predict([[6.5]])
print(y_pred_lin)
#Wrong prediction

#predict polynomial prediction
y_pred_pol = lin_regressor_2.predict(pol_features.fit_transform([[6.5]]))
print(y_pred_pol)

#more the degreee, more the overfitting.

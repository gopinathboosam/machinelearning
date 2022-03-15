# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:04:54 2021

@author: BoosamG
"""

#SouthGermanyCredit Regression

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

#Import Dataset
#Read data from asc file
# =============================================================================
# x = np.genfromtxt("explore/SouthGermanCredit/SouthGermanCredit.asc", dtype=list)
# print(x[0])
# =============================================================================

dataset = pd.read_csv("explore/SouthGermanCredit/SouthGermanCredit.asc", delimiter = ' ')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


def linearRegression(a,b):
    regressor = LinearRegression()
    regressor.fit(a, b)
    return regressor

def polynomialRegression():
    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(X_train)
    regressor = linearRegression(X_poly,y_train)
    return regressor

def decisionTreeRegression():
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(X_train, y_train)
    return regressor

def randomForestRegression():
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)
    return regressor

def predictResult(reg):
    y_pred = reg.predict(X_test)
    np.set_printoptions(precision=2)
    #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    return y_pred

def getMetrics(y_pred):
    return r2_score(y_test, y_pred)

regressor = randomForestRegression()
y_pred = predictResult(regressor)
r2 = getMetrics(y_pred)
print(r2)
    
    
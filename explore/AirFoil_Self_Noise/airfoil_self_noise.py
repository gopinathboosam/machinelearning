# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:58:08 2021

@author: BoosamG
"""
#AirFoil Self Noise Analysys

#import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

#Read .dat file content and convert it to dataframe
airfoildata = [i.strip().split() for i in open('explore/AirFoil_Self_Noise/airfoil_self_noise.dat').readlines()]
airfoildataframe = pd.DataFrame(airfoildata)
x = airfoildataframe.iloc[:,:-1].values
y = airfoildataframe.iloc[:,-1].values


#Splitting the data into train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



def linearRegression():
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    return regressor
    
def polynomialRegression():
    x_poly = poly_reg.fit_transform(x_train)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    return regressor

def decisiontreeRegression():
    regressor = DecisionTreeRegressor(random_state = 0)
    regressor.fit(x_train, y_train)
    return regressor

def randomforestRegression():
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(x_train, y_train)
    return regressor
    
def predictResult(regressor):
    y_pred = regressor.predict(x_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    return y_pred
    
def getRSquareScore(y_pred):
    return r2_score(y_test, y_pred)

def getMeanAbsoluteError(y_pred):
    return mean_absolute_error(y_pred, y_test)

regressor = decisiontreeRegression()
y_pred = predictResult(regressor)
print(getRSquareScore(y_pred))
print(getMeanAbsoluteError(y_pred))
# =============================================================================
# 0.9291663496248235
# 1.3490538205980074
# =============================================================================

#Randome Forest  yields best results with 0.929 as accuracy
# =============================================================================
# 
# import csv
# with open("explore/AirFoil_Self_Noise/airfoil_self_noise.csv", "wb") as f:
#     writer = csv.writer(f)
#     writer.writerows(airfoildata)
# 
# 
# import numpy as np
# data = np.loadtxt( 'explore/AirFoil_Self_Noise/airfoil_self_noise.dat' )
# print(data)
# =============================================================================

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:58:08 2021

@author: BoosamG
"""
#AirFoil Self Noise Analysys

#import libraries
import pandas as pd
import numpy as np


#Read .dat file content and convert it to dataframe
airfoildata = [i.strip().split() for i in open('explore/AirFoil_Self_Noise/airfoil_self_noise.dat').readlines()]
airfoildataframe = pd.DataFrame(airfoildata)
X = airfoildataframe.iloc[:,:-1].values
y = airfoildataframe.iloc[:,-1].values
y = y.reshape(len(y),1)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# Training the SVR model on the Training set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

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

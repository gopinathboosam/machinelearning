# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 15:29:25 2021

@author: BoosamG
"""

#ANN with Classification - Deep Learning
#Part 1
#Import Libraries:
import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

#Data preprocessing
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:-1].values#Gives values except last column
y = dataset.iloc[:,-1].values#Only takes last column

#Encoding Categorical Data
#Encoding using Lable Encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

#Encode using OHE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x[1,:])

#Split the train and test data
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.2, random_state = 1)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

#Part 2
#Building an ANN
#Initialize ANN
ann = tf.keras.models.Sequential()
#Add Input and First Layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#Add Hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Add Output Layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#Part 3
#Training ANN with Dataset
#Compile ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Train the dataset with the model
ann.fit(xtrain, ytrain, batch_size=32, epochs=100)

#Part 4
#Predcting the results
y_pred= ann.predict(xtest)
y_pred = (y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),ytest.reshape(len(ytest),1)),axis = 1))#Concatenating vertically

#Making confusion matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(ytest,y_pred)
print(cm)
accuracy_score(ytest,y_pred)

#Predict New Result
print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))>0.5)
print(ann.predict(sc.transform([[1,0,0,222,1,23,3,100,3,0,0,50000]])))

#Data_Preprocessing
#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Import Dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values#Gives values except last column
y = dataset.iloc[:,-1].values#Only takes last column

print(x)
print(y)


#Handle the missing data
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:,1:3] = imputer.fit_transform(x[:,1:3])

print(x)


#Encode the Categorical Data
#Encode Independant Variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
#d = ct.fit_transform(x)
#print(type(d))
x = np.array(ct.fit_transform(x))
print(x)

#Encode Dependant Data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)


#Splitting the dataset into trainingset and testset
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.2, random_state = 1)
print('***************************')
print(xtrain)
print('***************************')
print(xtest)
print('***************************')
print(ytrain)
print('***************************')
print(ytest)
print('***************************')


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain[:,3:] = sc.fit_transform(xtrain[:,3:])
xtest[:,3:] = sc.transform(xtest[:,3:])

print(xtrain)
print('***************************')
print(xtest)
print('***************************')
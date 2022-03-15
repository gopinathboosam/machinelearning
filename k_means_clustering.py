# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:48:06 2020

@author: BoosamG
"""

#K Means clustering

#import libraries
import numpy as np 
import matplotlib.pyplot as pt
import pandas as pd

#import dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values #Considering only last two features as variables


#Plot elbow method to identify optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters=i, init="k-means++", random_state=1)
    Kmeans.fit(x)
    wcss.append(Kmeans.inertia_)
pt.plot(range(1,11), wcss)
pt.title("Elbow Method")
pt.xlabel("Number of clusters")
pt.ylabel("WCSS")
pt.show()

#Train Kmeans model on dataset
#Use the identified number of clusters   which is 5 here
Kmeans = KMeans(n_clusters=5, init="k-means++", random_state=1)
y_pred = Kmeans.fit_predict(x) #Returns dependent variable
print(y_pred)


#visualizing clusters - Using the 2 features
pt.scatter(x[y_pred==0,0], x[y_pred==0,1], s=100, c='red', label = "Cluster 0")
pt.scatter(x[y_pred==1,0], x[y_pred==1,1], s=100, c='green', label = "Cluster 1")
pt.scatter(x[y_pred==2,0], x[y_pred==2,1], s=100, c='blue', label = "Cluster 2")
pt.scatter(x[y_pred==3,0], x[y_pred==3,1], s=100, c='magenta', label = "Cluster 3")
pt.scatter(x[y_pred==4,0], x[y_pred==4,1], s=100, c='black', label = "Cluster 4")
pt.scatter(Kmeans.cluster_centers_[:,0], Kmeans.cluster_centers_[:,1], s=400, c= 'yellow', label = "Centroids")
pt.title("Clusters")
pt.xlabel("Annual Income")
pt.ylabel("Spending")
pt.legend()
pt.show()
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 18:44:47 2020

@author: BoosamG
"""

#Hierarchical clustering

#import libraries
import numpy as np 
import matplotlib.pyplot as pt
import pandas as pd

#import dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values #Considering only last two features as variables

#Use Dendogram to identify optimal number of clusters
from scipy.cluster import hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))#minimum variance inside clusters
pt.title("Dendogram")
pt.xlabel("Customers")
pt.ylabel("Eucledian Distances")
pt.show()
#Find the optimal number of clusters by checking the plotted graph manually

#Train the model with number of clusters - which is 5 here
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage='ward')
y_pred = hc.fit_predict(x)
print(y_pred)

#Visualizing the clusters
pt.scatter(x[y_pred==0,0], x[y_pred==0,1], s=100, c='red', label = "Cluster 0")
pt.scatter(x[y_pred==1,0], x[y_pred==1,1], s=100, c='green', label = "Cluster 1")
pt.scatter(x[y_pred==2,0], x[y_pred==2,1], s=100, c='blue', label = "Cluster 2")
pt.scatter(x[y_pred==3,0], x[y_pred==3,1], s=100, c='magenta', label = "Cluster 3")
pt.scatter(x[y_pred==4,0], x[y_pred==4,1], s=100, c='black', label = "Cluster 4")
pt.title("Clusters")
pt.xlabel("Annual Income")
pt.ylabel("Spending")
pt.legend()
pt.show()
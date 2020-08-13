# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:01:25 2020

@author: pulki
"""


#Topic: Assignment - Clustering - mtcars
#-----------------------------
#libraries
#pip install kneed
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from pydataset import data
mtcars = data('mtcars')
data = mtcars.copy()
data
data.columns
k=2
#need for scaling : height & weight are in different scales
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)
scaled_features[:5]  #values between -3 to +3

kmeans = KMeans( init = 'random', n_clusters=2 , n_init=3, max_iter=300, random_state=42)
kmeans
kmeans.fit(scaled_features)
kmeans.inertia_
kmeans.cluster_centers_  #average or rep values
kmeans.n_iter_  #in 6 times, clusters stabilised
kmeans.labels_[:5]
kmeans.cluster_centers_.shape
kmeans.cluster_centers_[0:1]
kmeans.predict(scaled_features)
scaled_features[1:5]
scaled_features.columns
import pandas as pd
y=pd.DataFrame(scaled_features)
y
y.columns

clusterNos=kmeans.labels_
clusterNos

#mean of mpg, hp, wt
data.groupby(clusterNos).agg({'mpg':'mean','hp':'mean','wt':'mean'})

##plot scatter wt vs mpg with color cluster
plt.scatter(data.wt, data.mpg, c=clusterNos)
plt.xlabel('Weight')
plt.ylabel('Mileage')
plt.title('Color Cluster')
plt.show();

##plot scatter wt vs hp with color cluster
data.columns
plt.scatter(data.hp, data.wt, c=clusterNos)
plt.xlabel('Horse Power')
plt.ylabel('Weight')
plt.title('Color Cluster')
plt.show();

##plot scatter mpg vs hp with color cluster
plt.scatter(data.mpg, data.hp, c=clusterNos)
plt.xlabel('MPG')
plt.ylabel('HP')
plt.title('Color Cluster')
plt.show();
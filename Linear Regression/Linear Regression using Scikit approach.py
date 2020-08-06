# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:15:09 2020

@author: pulkit
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)


#https://s3.amazonaws.com/acadgildsite/wordpress_images/datasets/slr06/slr06.xls
data=pd.read_excel('https://s3.amazonaws.com/acadgildsite/wordpress_images/datasets/slr06/slr06.xls')
data.values
print(data.shape)
data.head().values

X = data.iloc[:,0].values
X
Y = data.iloc[:,1].values
Y
m = len(X)
m
# Import libraries and tools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Cannot use Rank 1 matrix in scikit learn
X = X.reshape((m, 1))
# Creating Model
reg = LinearRegression()
# Fitting training data
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)
# Calculating RMSE and  Score
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(X, Y)
print(np.sqrt(mse))
print(r2_score)


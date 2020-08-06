# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:25:29 2020

@author: pulkit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D

data=pd.read_csv('E:/analytics/datasets/student.csv')
data.shape
data.head()

# We will get scores to an array.
math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values
# Ploting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(math, read, write, color='#ef1234')
ax.set_xlabel('math')
ax.set_ylabel('read')
ax.set_zlabel('write')
ax.set_title(r'3D plot of features')
plt.show()


X = np.array([math, read]).T
X
Y = np.array(write).T

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Model Intialization
reg = LinearRegression()
# Data Fitting
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)
# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)
print(rmse)
print(r2)

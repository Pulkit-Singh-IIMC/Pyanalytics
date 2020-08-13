# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:30:55 2020

@author: pulki
"""


%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)

data=pd.read_csv('E:/analytics/datasets/headbrain.csv')
data.shape
data.head()

X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values

ax=plt.scatter(X,Y)
plt.line(X,Y)
plt.title('Linear Regression')
plt.xlabel("Head Size")
plt.ylabel('Brain Weight')

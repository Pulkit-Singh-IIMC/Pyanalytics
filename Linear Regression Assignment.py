# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:41:13 2020

@author: pulkit
"""


#%% Linear Regression -1 Marketing Data - Sales - YT, FB, print
#libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model #1st method
import statsmodels.api as sm  #2nd method
import matplotlib.pyplot as plt
import seaborn as sns

url ='https://raw.githubusercontent.com/DUanalytics/datasets/master/R/marketing.csv'
marketing = pd.read_csv(url)
marketing.head()

#describe data
marketing.describe()
marketing.shape
marketing.info()
X = marketing[['youtube','facebook','newspaper']] # here we have 3 independent variables for multiple regression. 
y = marketing['sales']

pairs(model)
#visualise few plots to check correlation
sns.scatterplot(data=marketing, x='youtube', y='sales')
sns.scatterplot(data=marketing, x='facebook', y='sales')
sns.scatterplot(data=marketing, x='newspaper', y='sales')

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size= 0.30, random_state=42)
X_train
y_train
X_test
y_test
X_test.shape
X_train.shape
y_train.shape
y_test.shape

#build the model
from sklearn.linear_model import LinearRegression
model = LinearRegression() 
model.fit(X_train,y_train)  

#predict on test values
y_pred = model.predict(X_test)
y_pred
y_pred2 = model.intercept_ + np.sum(model.coef_ * X_test, axis=1)
y_pred2
y_pred - y_pred2

#find metrics - R2, Adjt R2, RMSE, MAPE etc
model = sm.OLS(y_train, X_train).fit()
predictions = model.predict(X_test) 
print_model = model.summary()
print(print_model)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)
import math
math.sqrt(mean_squared_error(y_test,y_pred))
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred, multioutput="raw_values")


#predict on new value
newdata = pd.DataFrame({'youtube':[50,60,70], 'facebook':[20, 30, 40], 'newspaper':[70,75,80]})
newdata
y_new = model.predict(newdata)
y_new
#your ans should be close to [ 9.51, 11.85, 14.18] 

#conclude by few lines




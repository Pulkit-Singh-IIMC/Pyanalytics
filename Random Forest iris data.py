# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:00:23 2020

@author: pulki
"""


import numpy as np
import pandas as pd
from pydataset import data
from sklearn import datasets
import matplotlib.pyplot as plt
iris=datasets.load_iris()
X=iris.data[:,:]
y=iris.target
X, y
labels=iris.feature_names
labels

df=pd.DataFrame(iris.data)
df.columns=['sepal_length_(cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
df['target']=pd.Series(iris.target)
df

#%%plots #barplot
df.target.value_counts().plot.bar() 
plt.show();

#histogram 
df.plot.hist() 
plt.show();

df.plot.hist() 
plt.show();

#density 
df.plot.density() 
plt.show();

#correlation  plot 
import seaborn as sns
corr  =  df.corr()
sns.heatmap(corr,  annot=True) 
plt.show();

#%%
#train test split
from sklearn.model_selection import train_test_split
X_train,  X_test,  y_train,  y_test  =  train_test_split(X,  y,  test_size=0.3, random_state=1)
X_train.shape,  X_test.shape

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)

model.score(X_test,y_test)
y_predicted=model.predict(X_test)
y_predicted

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predicted)
cm

sns.heatmap(cm,  annot=True) 
plt.show();
df.columns
newdata=pd.DataFrame({'sepal_length_(cm)':[1,10], 'sepal width (cm)':[2,4], 'petal length (cm)':[5,9],'petal width (cm)':[1,2]})
newdata
df.describe()

model.predict(newdata)

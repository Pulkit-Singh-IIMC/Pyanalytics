# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 19:01:03 2020

@author: pulki
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

suv=pd.read_csv('E:/analytics/datasets/suv.csv')
suv.head(10)

#data wrangling (removing NaN values)
suv.isnull()
suv.isnull().sum()

sns.heatmap(suv.isnull())
suv
gender=pd.get_dummies(suv['Gender'],drop_first=True)
gender
suv=pd.concat([suv,gender],axis=1)
suv.drop(['Gender'],axis=1,inplace=True)
suv
X=suv.iloc[:,[1,2,4]].values
X
y=suv.iloc[:,3].values
y

#train vs test data
from sklearn.model_selection import train_test_split
train_test_split?
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

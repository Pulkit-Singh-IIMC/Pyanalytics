# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 16:54:09 2020

@author: pulki
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

titanic=pd.read_csv('E:/analytics/datasets/titanic.csv')
titanic.head(10)

#plotting graphs
sns.countplot(x='Survived',data=titanic)
sns.countplot(x='Survived',hue="Sex", data=titanic)
sns.countplot(x='Survived',hue='Pclass',data=titanic)
titanic['Age'].plot.hist()
titanic['Fare'].plot.hist(bins=20,figsize=(10,5))
sns.countplot(x='SibSp',data=titanic)

#data wrangling (removing NaN values)
titanic.isnull()
titanic.isnull().sum()

sns.heatmap(titanic.isnull())
sns.boxplot(x='Pclass',y='Age',hue=None,data=titanic)
titanic.head()
titanic.drop('Cabin',axis=1,inplace=True)
titanic.dropna(inplace=True)
titanic.head()
sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False)
titanic.isnull().sum()


sex= pd.get_dummies(titanic['Sex'],drop_first=True)
embark=pd.get_dummies(titanic['Embarked'],drop_first=True)
embark.head()
Pcl=pd.get_dummies(titanic['Pclass'],drop_first=True)
Pcl.head()
titanic=pd.concat([titanic,sex,embark,Pcl],axis=1)
titanic.head()
titanic.drop(['Sex','Embarked','PassengerId',"Name",'Ticket'],axis=1,inplace=True)
titanic.head()
titanic.drop(['Pclass'],inplace=True,axis=1)
titanic.head()
y=titanic['Survived']
X=titanic.drop('Survived', axis=1)
X
y

#train vs test data
from sklearn.model_selection import train_test_split
train_test_split?
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

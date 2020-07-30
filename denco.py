# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 12:32:59 2020

@author: pulkit
"""


import numpy as np
import pandas as pd

data= pd.read_csv('E:/analytics/datasets/denco.csv')
data.head()
data.columns
data.shape
data.dtypes
data.describe()
data.index

#most loyal customer
grouped=data.custname.value_counts()
grouped
grouped.head(5)

#Which customers contribute the most to their revenue
grouped_rev=data.groupby('custname').agg({'revenue':sum})
grouped_rev.sort_values('revenue',ascending=False).head()

#What part numbers bring in to significant portion of revenue
grouped_partnum=data.groupby('partnum').agg({'revenue':sum})
grouped_partnum.sort_values('revenue',ascending=False).head()

#What parts have the highest profit margin ?
grouped_pm=data.groupby('partnum').agg({'margin':sum})
grouped_pm.sort_values('margin',ascending=False).head()

#Who are their top buying customers
grouped.head(1)

#Who are the customers who are bringing more revenue
grouped_rev.head(1)

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:31:26 2020

@author: pulkit
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import time
import logging
pd.set_option('display.max_columns',None)

dataset = [['Apple', 'Beer', 'Rice', 'Chicken'],  ['Apple', 'Beer', 'Rice'], ['Apple', 'Beer'],  ['Apple', 'Bananas'], ['Milk', 'Beer', 'Rice', 'Chicken'], ['Milk', 'Beer', 'Rice'],  ['Milk', 'Beer'], ['Apple', 'Bananas']]

dataset
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
te_ary
df = pd.DataFrame(te_ary, columns=te.columns_)
df

#%%% #frequent itemsets 
support_threshold = 0.3

frequent_itemsets = apriori(df, min_support= support_threshold, use_colnames = True)
frequent_itemsets
frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
frequent_itemsets
frequent_itemsets.head(2)
#%%%%  - Support Rules

supportRules1 = association_rules(frequent_itemsets, metric="support", min_threshold = 0.3)
print(supportRules1)

print(supportRules1[['antecedents', 'consequents', 'support','confidence','lift']])
supportRules1[(supportRules1.confidence > .5)  & (supportRules1.lift > 1)]
#%%%% Lift  : generally > 1 for strong associations

lift1 = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(lift1)
lift1
print(lift1[['antecedents', 'consequents', 'support', 'lift','confidence']])

#twin condition : lift> 2;  confidence > .5, support > .2
lift1[(lift1.confidence > .5)  & (lift1.support > 0.2)]


#%%%% Confidence

confidence1 = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print(confidence1)

print(confidence1[['antecedents', 'consequents', 'support','confidence','lift']])

confidence1[(confidence1.support > .2)  & (confidence1.lift > 1)]


#%%
#Marketing Strategy

#  antecedents consequents  
#0      (Beer)      (Rice)
#1      (Rice)      (Beer)
#3      (Milk)      (Beer)

#For rule 2 & 3, confidence level is 1 which means out of all the transactions that contain rice or milk, 100% of the transactions also contain beer and the lift is 1.33. So rice, milk and beer should neither be kept together in the outlet nor they should be bundled together.

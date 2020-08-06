# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 12:54:12 2020

@author: pulki
"""


dog_dict={'name':'Freddie','age':9,'is_vaccinated':True,'height':1.1,'belongings':['bone','ball'],'birth_year':2001}
dog_dict
dog_dict['name']

import pandas as pd
people_dict={'weight':pd.Series([68,83,112],index=['alice','bob','charles']),'birthyear':pd.Series([1984,1985,1992],index=["bob",'alice','charles']),'children':pd.Series([0,2],index=["charles",'bob']),'hobby':pd.Series(['biking','dancing'],index=['alice','bob'])}
pd.DataFrame(people_dict).describe()
import matplotlib.pyplot as plt
plt.plot(pd.DataFrame(people_dict))

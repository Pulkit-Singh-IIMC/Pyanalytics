# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:43:41 2020

@author: pulki
"""


#Topic:  Decision  tree  using  Wine  Data  Set # 	
#libraries
import  pandas  as  pd 
import  numpy  as  np
import  matplotlib.pyplot  as  plt 
import  seaborn  as  sns
from  sklearn.model_selection  import  train_test_split

#dataset  -  understand  it  first  :  target  class  based  on  other  parameters 
#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html
from  sklearn.datasets  import  load_wine 
wine  =  load_wine()
wine 
wine.data

#  feature  matrix 
X  =  wine.data
X
#  target  vector 
y  =  wine.target 
y
#  class  labels
labels  =  wine.feature_names 
labels

#join  X  &  y  to  dataframe
df  =  pd.DataFrame(wine.data) #name  the  columns
df.columns  =  ['Alcohol',  'Malicacid',  'Ash',  'AlcalinityOfAsh',  'Magnesium', 'TotalPhenols',  'Flavanoids',  'NonflavanoidPhenols',  'Proanthocyanins', 'ColorIntensity',  'Hue',  'OD280_OD315',  'Proline']
df
df['target']  =  pd.Series(wine.target)  #add  class  column  as  target df
df
#%%%  what  to  do  in  this  assignment
##in  this  case  example  Alcohol,  Magnesium,  Proline  are  used  as  IV  to  predict  two target  class,  you  required  to  select  3  different  sets  of  variables  and  then predict  the  class
df.target.value_counts() 
data1  =  df[df.target  !=  2] 
data1.target.value_counts()

from sklearn import tree
m1 = data1[['Malicacid','Ash','AlcalinityOfAsh','target']]
m1.target.value_counts()
m1.head()
data1.describe()

newdata1  =  pd.DataFrame({'Malicacid':[3,5],  'Ash':[2,3],  'AlcalinityOfAsh':[14, 21]})

X1  =  m1[['Malicacid',  'Ash',  'AlcalinityOfAsh']].values 
X1
y1 =  m1['target'].values 
y1

m1.columns 
#train  and  test
X1_train,  X1_test,  y1_train,  y1_test  =  train_test_split(X1,  y1,  test_size=0.3, random_state=1)
X1_train.shape,  X1_test.shape

from  sklearn  import  tree
clsModel1  =  tree.DecisionTreeClassifier()	#model  with  parameter 
clsModel1.fit(X1_train,  y1_train)

#predict

y1_pred  =  clsModel1.predict(X1_test) 
len(y1_pred)

#%%%performance
from  sklearn  import  metrics 
metrics.accuracy_score(y1_test,  y1_pred)	
#accuracy
from  sklearn.metrics  import  confusion_matrix 
confusion_matrix  =  confusion_matrix(y1_test,  y1_pred)
 
print(confusion_matrix)

from  sklearn.metrics  import  classification_report 
print(classification_report(y1_test,  y1_pred))

#predict  on  unknown  target newdata
y1_pred1  =  clsModel1.predict(newdata1) 
y1_pred1
y1_pred1C  =  clsModel1.predict_proba(newdata1) 
y1_pred1C

pd.concat([newdata1,  pd.Series(y1_pred1)],  axis=1) 
#first  predicted  as  1,  2nd  rows  as  1

#visualise  the  tree
from  sklearn  import  tree
tree.plot_tree(decision_tree=clsModel1,  node_ids=True,  filled=True  ) 
fig  =  plt.figure(figsize=(10,8))
_  =  tree.plot_tree(clsModel1,  feature_names=  ['Malicacid',  'Ash',  'AlcalinityOfAsh'], class_names=['0','1'],  filled=True)	#see  plot

#%%
#Model 2
df.columns
df.target.value_counts() 
data2  =  df[df.target  !=  2] 
data2.target.value_counts()

from sklearn import tree
m2 = data2[['Malicacid','ColorIntensity','OD280_OD315','target']]
m2.target.value_counts()
m2.head()
data2.describe()
data2[['Malicacid','ColorIntensity','OD280_OD315','target']].describe()
newdata2  =  pd.DataFrame({'Malicacid':[1,5],  'ColorIntensity':[2,7],  'OD280_OD315':[1.7, 3.8]})

X2  =  m2[['Malicacid','ColorIntensity','OD280_OD315']].values 
X2
y2 =  m1['target'].values 
y2

m2.columns 
#train  and  test
X2_train,  X2_test,  y2_train,  y2_test  =  train_test_split(X2,  y2,  test_size=0.3, random_state=1)
X2_train.shape,  X2_test.shape

from  sklearn  import  tree
clsModel2  =  tree.DecisionTreeClassifier()	#model  with  parameter 
clsModel2.fit(X2_train,  y2_train)

#predict

y2_pred  =  clsModel2.predict(X2_test) 
len(y2_pred)

#%%%performance
from  sklearn  import  metrics 
metrics.accuracy_score(y2_test,  y2_pred)	
#accuracy
from  sklearn.metrics  import  confusion_matrix 
confusion_matrix2  =  confusion_matrix(y2_test,  y2_pred)
 
print(confusion_matrix2)

from  sklearn.metrics  import  classification_report 
print(classification_report(y2_test,  y2_pred))

#predict  on  unknown  target newdata
y2_pred2  =  clsModel2.predict(newdata2) 
y2_pred2
y2_pred2C  =  clsModel2.predict_proba(newdata2) 
y2_pred2C

pd.concat([newdata2,  pd.Series(y2_pred2)],  axis=1) 
#first  predicted  as  1,  2nd  rows  as  0

#visualise  the  tree
from  sklearn  import  tree
tree.plot_tree(decision_tree=clsModel2,  node_ids=True,  filled=True  ) 
fig  =  plt.figure(figsize=(10,8))
_  =  tree.plot_tree(clsModel2,  feature_names=  ['Malicacid',  'ColorIntensity',  'OD280_OD315'], class_names=['0','1'],  filled=True)	#see  plot

#you  will  have  use  different  variables,  create  test  data  accordingly  for prediction

#%%%
#select  class  0  &  1  from  target  column df.shape
df.target.value_counts() 
data  =  df[df.target  !=  2] 
data.target.value_counts()

#use  only  3  columns  for  constructing  a  model
data2  =  data[['Alcohol',  'Magnesium',  'Proline',  'target']] 
data2.head()
 
data2.describe()

#%%plots #barplot
data2.target.value_counts().plot.bar() 
plt.show();

#histogram 
data2.Alcohol.plot.hist() 
plt.show();

#density 
data2.Alcohol.plot.density() 
plt.show();

#correlation  plot 
corr  =  data2.corr()
sns.heatmap(corr,  annot=True) 
plt.show();

#%%training vs testing for Random Forest

df
X  =  df.drop(['target'], axis=1).values 
X
y  =  df['target'].values 
y
df.columns 
#train  and  test
X_train,  X_test,  y_train,  y_test  =  train_test_split(X,  y,  test_size=0.3, random_state=1)
X_train.shape,  X_test.shape
print(X_train)
y_train.shape, y_test.shape
#%%
#Random Forest modeling

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)

model.score(X_test,y_test)
y_predicted=model.predict(X_test)
y_predicted

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_predicted)
cm

df['OD280_OD315'].describe()
newdata2  =  pd.DataFrame({'Malicacid':[1,5],  'ColorIntensity':[2,7],  'OD280_OD315':[1.7, 3.8]})
df.columns
X_test
newdata=pd.DataFrame({'Alcohol':[12,13],  'Malicacid':[1,4],  'Ash':[1,2],  'AlcalinityOfAsh':[14,23],  'Magnesium':[72,100], 'TotalPhenols':[1,3],  'Flavanoids':[1,4],  'NonflavanoidPhenols':[0.4,0.57],  'Proanthocyanins':[0.4,1.9], 'ColorIntensity':[4,10],  'Hue':[0.8,1.4],  'OD280_OD315':[1,4],  'Proline':[300,1000]})

newdata  =  pd.DataFrame({'Alcohol':[12,13],  'Malicacid':[1.3,4.7],  'Ash':[1.9,2.6],  'AlcalinityOfAsh':[13.9,23.5],  'Magnesium':[72,100],  'TotalPhenols':[1.3,2.9],  'Flavanoids':[0.7,3.9],  'NonflavanoidPhenols':[0.4,0.57],  'Proanthocyanins':[0.4,1.9],  'ColorIntensity':[3.9,9.9],  'Hue':[0.8,1.4],  'OD280_OD315':[1.4,3.6],  'Proline':[300,1000]})
newdata
model.predict(newdata)


#%%decision  tree  model 
from  sklearn  import  tree
clsModel  =  tree.DecisionTreeClassifier()	#model  with  parameter 
clsModel.fit(X_train,  y_train)

#predict
y_pred1  =  clsModel.predict(X_test) 
len(y_pred1)

#%%%performance
from  sklearn  import  metrics 
metrics.accuracy_score(y_test,  y_pred1)	
#accuracy
from  sklearn.metrics  import  confusion_matrix 
confusion_matrix  =  confusion_matrix(y_test,  y_pred1)
 
print(confusion_matrix)

from  sklearn.metrics  import  classification_report 
print(classification_report(y_test,  y_pred1))

#predict  on  unknown  target newdata
y_pred1B  =  clsModel.predict(newdata) 
y_pred1B
y_pred1C  =  clsModel.predict_proba(newdata) 
y_pred1C

pd.concat([newdata,  pd.Series(y_pred1B)],  axis=1) 
#first  predicted  as  1,  2nd  rows  as  0

#visualise  the  tree
from  sklearn  import  tree
tree.plot_tree(decision_tree=clsModel,  node_ids=True,  filled=True  ) 
fig  =  plt.figure(figsize=(10,8))
_  =  tree.plot_tree(clsModel,  feature_names=  ['Alcohol',  'Magnesium',  'Proline'], class_names=['0','1'],  filled=True)	#see  plot

#explain  any  decision  node(not  root,  not  leaf)
# For model 2, decision node Malicacid<=1.24 gini=0.245 samples 49 value=[42,7] class =0. The node gets divided into 2 nodes. It ends for False and further grows for True. The true side of the node can be read as follows. For colorintensity >3.46, OD280_OD315>2.505, Malic acid>1.24 and colorintensity<=3.945, 36 samples on the true side have 0 class and 10 samples on False side also have 0 class and further gets divided into nodes. 


#Mention  2  applications  of  Decision  Tree  (domain  and  what  can  we  predict) 
#When there are more than one courses of action decision tree can help to easily visualise the decision making 
#For each alternative in a decision tree there are pros and cons
#end here

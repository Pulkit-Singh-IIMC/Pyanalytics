#Case Study on mtcars dataset in Python	download data

#Download data
import statsmodels.api as sm
#https://vincentarelbundock.github.io/Rdatasets/datasets.html
dataset_mtcars = sm.datasets.get_rdataset(dataname='mtcars', package='datasets')
dataset_mtcars.data.head()
mtcars = dataset_mtcars.data
#structure
mtcars.shape

#summary
import pandas as pd
mtcars.describe()

#print first / last few rows
mtcars.iloc[[0,1,2,3,28,29,30,31],:]

#print number of rows
mtcars['mpg'].count()

#number of columns
len(mtcars.columns)

#print names of columns
mtcars.columns

#Filter Rows
#cars with cyl=8
mtcars[mtcars['cyl']==8]
#cars with mpg <= 27
mtcars[mtcars['mpg']<=27]
#rows match auto tx
mtcars[mtcars['am']==0]
#First Row
mtcars.head(1)
mtcars.iloc[[0]]
#last Row
mtcars.tail(1)

# 1st, 4th, 7th, 25th row + 1st 6th 7th columns.
mtcars.iloc[[0,3,6,24],[0,5,6]]
mtcars.columns

# first 5 rows and 5th, 6th, 7th columns of data frame
mtcars.iloc[:5,[4,5,6]]
mtcars.head()
#rows between 25 and 3rd last
mtcars.iloc[25:-3]
mtcars.tail(10)

#alternative rows and alternative column
mtcars.iloc[::2,::2]
mtcars.shape
#find row with Mazda RX4 Wag and columns cyl, am
mtcars.loc[['Mazda RX4 Wag'],['cyl','am']]

#find row betwee Merc 280 and Volvo 142E Mazda RX4 Wag and columns cyl, am
mtcars.loc['Merc 280':'Volvo 142E',['cyl','am']]

# mpg > 23 or wt < 2
mtcars.loc[(mtcars['mpg'] > 23) | (mtcars['wt'] < 2)]

#using lambda for above


#with or condition

#find unique rows of cyl, am, gear


#create new columns: first make a copy of mtcars to mtcars2

#keeps other cols and divide displacement by 61

# multiple mpg * 1.5 and save as original column


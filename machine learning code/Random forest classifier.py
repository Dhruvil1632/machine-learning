#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error


# In[2]:


data = pd.read_csv("C:/Users/dhruv/Desktop/python sample file/diabetes.csv")
data.head()
print(len(data))


# In[3]:


data.head()


# In[4]:


clas = LabelEncoder()
target= clas.fit_transform(data['class'])
data.drop(['class'],axis='columns',inplace=True)


# In[5]:


x_train = data[0:650]
x_test = data[650:]
y_train = target[0:650]
y_test = target[650:]
print(len(x_train))
print(len(x_test))


# In[6]:


model = RandomForestClassifier(n_estimators=10,warm_start=True,criterion="gini")
model.fit(x_train , y_train)


# In[7]:


y_predicted = model.predict(x_test)
count=0
for i in range(0,len(x_test)):
    if y_predicted[i] == y_test[i]:
        count = count+1
print("mached data with target " , count)
print("length of testing data" , len(x_test))
print(mean_squared_error(y_test,y_predicted))












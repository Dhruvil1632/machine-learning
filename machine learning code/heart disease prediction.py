#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


# In[2]:


data = pd.read_csv("C:/Users/dhruv/Desktop/python sample file/heart.csv",sep="\t",quotechar="/")
data.head()


# In[3]:


correlation = data.corr(method='pearson')
print(correlation)


# In[4]:


target = data['target']
data.drop(['target'],axis='columns',inplace=True)
x_train ,x_test ,y_train , y_test = train_test_split(data,target,test_size=0.08,random_state=4)


# In[5]:


print(len(x_train))
print(len(x_test))


# In[6]:


model = BernoulliNB()
model.fit(x_train , y_train)


# In[7]:


y_predicted = model.predict(x_test)


# In[8]:


print(mean_squared_error(y_test,y_predicted))
print(accuracy_score(y_test,y_predicted))


# In[9]:


count=0
for i in range(0,len(x_test)):
    if y_test.values[i]==y_predicted[i]:
        count = count+1
print(len(x_test))
print(count)


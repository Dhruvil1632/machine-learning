#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error , accuracy_score


# In[2]:


data = pd.read_csv("C:/Users/dhruv/Desktop/python sample file/creditcard.csv")
data.tail()


# In[3]:


len(data)


# In[4]:


target = data['Class']
data.drop(['Class'],axis='columns',inplace=True)


# In[5]:


x_train , x_test , y_train , y_test = train_test_split(data,target,test_size=0.1,random_state=3)


# In[6]:


len(x_test)


# In[11]:


model = BernoulliNB()
model.fit(x_train,y_train)


# In[12]:


y_predicted = model.predict(x_test)
print(mean_squared_error(y_predicted,y_test))


# In[13]:


print(accuracy_score(y_predicted,y_test))






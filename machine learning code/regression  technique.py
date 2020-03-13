#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression


# In[2]:


data = pd.read_csv("C:/Users/dhruv/Desktop/python sample file/cars.csv")
data.drop(['Make','Model','Type','Drive'],axis='columns',inplace=True)
data.tail()
#print(data.keys())


# In[3]:


data.info()


# In[4]:


correlation = data.corr(method='spearman')
print(correlation)


# In[5]:


le_size = LabelEncoder()
data['size'] = le_size.fit_transform(data['Size'])
target = data['size']
data.drop(['Size'],axis='columns',inplace=True)


# In[6]:


print(len(data))


# In[7]:


x_train , x_test , y_train , y_test = train_test_split(data,target,test_size=0.2,random_state=2)


# In[8]:


len(x_train)


# In[9]:


#linear regression
model = linear_model.LinearRegression()
model.fit(x_train , y_train)


# In[10]:


ly_predicted = model.predict(x_test)
print(mean_squared_error(y_test,ly_predicted))


# In[11]:


#ordinary least square regression
model_ols = sm.OLS(endog=y_train,exog=x_train).fit()
oy_predicted= model_ols.predict(x_test)
print(mean_squared_error(y_test,oy_predicted))


# In[12]:


data2 = pd.read_csv("C:/Users/dhruv/Desktop/python sample file/logistic.csv")
data2.head()


# In[13]:


#plt.figure(figsize=(10,7))
plt.scatter(data2['age'],data2['have_ins'])


# In[14]:


len(data2)


# In[15]:


x2_train = data2[['age']][0:10]
x2_test = data2[['age']][10:]
y2_train = data2['have_ins'][0:10]
y2_test = data2['have_ins'][10:]


# In[16]:


#logistic regression
logi_model = LogisticRegression()
logi_model.fit(x2_train,y2_train)


# In[18]:


y_predicted = logi_model.predict(x2_test)


# In[20]:


print(mean_squared_error(y_predicted , y2_test))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier , BaggingClassifier
from sklearn.metrics import mean_squared_error


# In[2]:


data = pd.read_csv("C:/Users/dhruv/Desktop/python sample file/diabetes.csv")
data.head()


# In[3]:


Class = LabelEncoder()
target = Class.fit_transform(data['class'])
data.drop(['class'],axis='columns',inplace=True)


# In[4]:


x_train  , x_test ,y_train , y_test = train_test_split(data,target,test_size=0.1,random_state=3)
print(len(x_train))
print(len(x_test))


# In[5]:


model = DecisionTreeClassifier(criterion="gini")
#model.fit(x_train,y_train)
boost = AdaBoostClassifier(base_estimator=model,n_estimators=250,learning_rate=1)
boost.fit(x_train , y_train)
y_predicted = boost.predict(x_test)
print(mean_squared_error(y_test,y_predicted))


# In[6]:


boost1 = BaggingClassifier(base_estimator=model,n_estimators=300)
boost1.fit(x_train ,y_train)
yb = boost1.predict(x_test)
print(mean_squared_error(y_test,yb))







#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error


# In[2]:


data= pd.read_csv("C:/Users/dhruv/Desktop/python sample file/winequality.csv",sep=";")
data.head()


# In[3]:


print(len(np.unique(data['quality'])))


# In[4]:


target = data['quality']
data.drop(['quality'],axis='columns',inplace=True)


# In[5]:


x_train ,x_test , y_train , y_test =  train_test_split(data,target,test_size=0.1,random_state=5)


# In[6]:


print(len(x_train))
print(len(x_test))


# In[7]:


#reg = LogisticRegression()
tree = DecisionTreeClassifier()
bayes = GaussianNB()
meta = KNeighborsClassifier(n_neighbors=10)


# In[8]:


model = StackingClassifier(classifiers=[tree,bayes],use_probas=True,meta_classifier=meta)
model.fit(x_train , y_train)


# In[9]:


y_predicted = model.predict(x_test)
print(mean_squared_error(y_test,y_predicted))







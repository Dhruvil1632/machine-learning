#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score , mean_squared_error


# In[2]:


data = pd.read_csv("C:/Users/dhruv/Desktop/python sample file/KNN.csv")
data.head()


# In[3]:


size = LabelEncoder()
target = size.fit_transform(data['T-shirt size'])
data.drop(['T-shirt size'],axis='columns',inplace=True)
x_train , x_test , y_train , y_test = train_test_split(data,target,test_size=0.1,random_state=3)


# In[4]:


print(len(x_train))
print(len(y_train))


# In[5]:


model = KNeighborsClassifier(n_neighbors=4)
model.fit(x_train ,y_train)
y_predicted = model.predict(x_test)
print(mean_squared_error(y_predicted , y_test))







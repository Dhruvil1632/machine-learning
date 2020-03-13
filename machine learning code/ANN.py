#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


sample = load_breast_cancer()
print(sample.keys())


# In[3]:


data = pd.DataFrame(sample['data'] , columns = sample['feature_names'])
data


# In[4]:


target = sample['target']


# In[5]:


x_train , x_test , y_train , y_test = train_test_split(data , target , test_size = 0.1 , random_state = 4)


# In[6]:


print(len(x_train))
print(len(y_test))


# In[7]:


classifier =  Sequential()


# In[8]:


#input and first hidden layer
classifier.add(Dense(output_dim= 15 ,activation = "relu" , init = "uniform" , input_dim = 30))


# In[9]:


# second hidden layer
classifier.add(Dense(output_dim = 7 , activation = "linear" , init = "uniform" ))
# third hidden layer
classifier.add(Dense(output_dim = 3 , activation = "linear" , init = "uniform"))
#final output layer
classifier.add(Dense(output_dim =1 , activation = "sigmoid"))


# In[10]:


classifier.compile(optimizer = "Adam"  , loss = "mean_squared_error" , metrics = ['accuracy'])


# In[11]:


classifier.fit(x_train , y_train , batch_size =32 , epochs = 200)


# In[12]:


y_predicted = classifier.predict(x_test)


# In[13]:


mean_squared_error(y_predicted , y_test)


# In[14]:


i=0
for i in range(0,len(x_test)):
    if y_predicted[i] == y_test[i]:
        i=i+1
print("matched target ",i) 
print("length of total testing" , len(y_test))       













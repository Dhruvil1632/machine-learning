#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import mean_squared_error


# In[2]:


data = pd.read_csv("spam.csv",encoding='latin-1')


# In[3]:


data.keys()
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis='columns',inplace=True)
data.head()


# In[4]:


le_status = LabelEncoder()
data['status'] = le_status.fit_transform(data['status'])
data.head()


# In[5]:


conv = CountVectorizer()
mess = conv.fit_transform(data['message'])


# In[6]:


len(conv.get_feature_names())


# In[7]:


new_mess = mess.toarray()
print(len(new_mess))


# In[8]:


x_train , x_test , y_train , y_test = train_test_split(new_mess , data['status'],test_size=0.2,random_state=4)
print(len(x_train))
print(len(x_test))


# In[9]:


#get back message
conv.inverse_transform(x_train[0])


# In[10]:


Gauss = GaussianNB()
Gauss.fit(x_train,y_train)


# In[11]:


y_predicted_g = Gauss.predict(x_test)
print(mean_squared_error(y_predicted_g,y_test))
print(y_test.values[1])


# In[12]:


Bern = BernoulliNB()
Bern.fit(x_train,y_train)


# In[13]:


y_predicted_b = Bern.predict(x_test)
print(mean_squared_error(y_predicted_b,y_test))


# In[14]:


Mult = MultinomialNB()
Mult.fit(x_train,y_train)


# In[15]:


y_predicted_m = Mult.predict(x_test)
print(mean_squared_error(y_predicted_m,y_test))


# In[16]:


g=0
b=0
m=0
for i in range(0,len(y_test)):
    if y_predicted_g[i]==y_test.values[i]:
        g=g+1
    if y_predicted_b[i]==y_test.values[i]:  
        b=b+1
    if y_predicted_m[i]==y_test.values[i]:  
        m=m+1
print("matching values of actual data and predicted data")        
print("g:",g)  
print("b:",b)
print("m:",m)








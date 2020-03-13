#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


data= pd.read_csv("C:/Users/dhruv/Desktop/python sample file/winequality.csv",sep=";")
data.head()


# In[3]:


correlation = data.corr(method='spearman')
print(correlation)


# In[4]:


len(np.unique(data['quality']))


# In[5]:


new_frame = pd.DataFrame()
new_frame['alcohol'] = data['alcohol']
new_frame['quality'] = data['quality']
new_frame.head()


# In[6]:


plt.figure(figsize=(10,7))
plt.scatter(new_frame['alcohol'],new_frame['quality'])


# In[7]:


print(len(new_frame))


# In[8]:


model = KMeans(n_clusters=6)
model.fit(new_frame)


# In[9]:


print(model.cluster_centers_)
print(model.labels_)


# In[10]:


print(np.unique(model.labels_))


# In[11]:


model.cluster_centers_


# In[12]:


model.cluster_centers_[1][0]


# In[13]:


for i in range(0,6):
    plt.scatter(model.cluster_centers_[i][0],model.cluster_centers_[i][1])


# In[17]:


new_frame['clustter'] = model.predict(new_frame)


# In[18]:


new_frame


# In[21]:


np.unique(new_frame['clustter'])


# In[25]:


plt.figure(figsize=(10,7 ))
for i in range(0,len(new_frame)):
    if new_frame['clustter'][i] == 0:
        plt.scatter(new_frame['alcohol'][i],new_frame['quality'][i],color='blue')
    elif new_frame['clustter'][i] == 1:
        plt.scatter(new_frame['alcohol'][i],new_frame['quality'][i],color='red') 
    elif new_frame['clustter'][i] == 2:
        plt.scatter(new_frame['alcohol'][i],new_frame['quality'][i],color='green')    
    elif new_frame['clustter'][i] == 3:
        plt.scatter(new_frame['alcohol'][i],new_frame['quality'][i],color='yellow')
    elif new_frame['clustter'][i] == 4:
        plt.scatter(new_frame['alcohol'][i],new_frame['quality'][i],color='orange')
    else:
        plt.scatter(new_frame['alcohol'][i],new_frame['quality'][i],color='black')
plt.legend()       








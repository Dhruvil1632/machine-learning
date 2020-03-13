#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
iris=load_iris()


# In[2]:


dir(iris)


# In[3]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)


# In[4]:


df.head()


# In[5]:


df['target']=iris.target
df


# In[6]:


iris.target_names


# In[7]:


setosa = df[df.target==0]
versicolor = df[df.target==1]
virgincia =  df[df.target==2]


# In[8]:


df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
df


# In[9]:


versicolor.head()


# In[10]:


setosa = df[df.target==0]
versicolor = df[df.target==1]
virgincia =  df[df.target==2]
plt.scatter(setosa['sepal length (cm)'],setosa['sepal width (cm)'],color="blue",label="setosa")
plt.scatter(versicolor ['sepal length (cm)'],versicolor ['sepal width (cm)'],color="green",label="versicolor")
plt.scatter(virgincia['sepal length (cm)'],virgincia['sepal width (cm)'],color="yellow",label="virgincia")
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()


# In[11]:


plt.scatter(setosa['petal length (cm)'],setosa['petal width (cm)'],color="blue",label="setosa")
plt.scatter(versicolor ['petal length (cm)'],versicolor ['petal width (cm)'],color="green",label="versicolor")
plt.scatter(virgincia['petal length (cm)'],virgincia['petal width (cm)'],color="yellow",label="virgincia")
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.legend()


# In[12]:


y=df.target
df.drop(['target','flower_name'],axis='columns',inplace=True)


# In[13]:


x_train ,x_test, y_train,y_test= train_test_split(df,y,test_size=0.2)


# In[14]:


print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))


# In[15]:


model=SVC(C=1.0 ,degree=3)


# In[16]:


model.fit(x_train , y_train)


# In[17]:


y_predicted = model.predict(x_test)
print(mean_squared_error(y_predicted , y_test))







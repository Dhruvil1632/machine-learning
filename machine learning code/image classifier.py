#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from sklearn.metrics import mean_squared_error


# In[2]:


location = "C:/Users/dhruv/Desktop/image"
type_ = ["train_cat/","train_dog/"]


# In[3]:


size=900


# In[4]:


data = list()
target = list()
for animal in type_:
    path = os.path.join(location,animal)
    img_class = type_.index(animal)
    for img in os.listdir(path):
        img_path= os.path.join(path,img)
        img_array = cv2.imread(img_path)
        resize_im = cv2.resize(img_array,(size,size))
        new_img = cv2.cvtColor(resize_im, cv2.COLOR_BGR2RGB)
        data.append(new_img)
        target.append(img_class)
  


# In[5]:


print(len(data))
print(len(target))


# In[ ]:





# In[6]:


x_train , x_test , y_train , y_test =  train_test_split(data , target , test_size=0.1 , random_state = 4)


# In[7]:


x_train = np.array(x_train)
x_test = np.array(x_test)
y_train  =  np.array(y_train)
y_test = np.array(y_test)


# In[8]:


x_train[1].shape


# In[9]:


classifier = Sequential()
# creating convolution layer
classifier.add(Conv2D(32,3,3,  input_shape = x_train[1].shape , activation ="relu" ))


# In[10]:


# pooling 
classifier.add(MaxPooling2D(pool_size = (2,2)))


# In[11]:


# creating convolution layer
classifier.add(Conv2D(32,3,3, activation ="relu" ))
# pooling 
classifier.add(MaxPooling2D(pool_size = (2,2)))


# In[12]:


#flattening
classifier.add(Flatten())


# In[13]:


#full connection
classifier.add(Dense(units=128 , activation ="relu"))
classifier.add(Dense(units = 1 , activation = "sigmoid"))


# In[14]:


classifier.compile(optimizer = "Adam" , loss = "binary_crossentropy" , metrics = ['accuracy'])


# In[15]:


classifier.fit(x_train , y_train , batch_size = 20 , epochs = 1)


# In[16]:


prediction =  classifier.predict(x_test)


# In[17]:


print(mean_squared_error(prediction , y_test))


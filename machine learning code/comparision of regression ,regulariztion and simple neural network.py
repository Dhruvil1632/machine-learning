#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statistics as s
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv("C:/Users/dhruv/Desktop/python sample file/pollution.csv")
#data.head()


# In[3]:


data.keys()


# In[4]:


new_frame = pd.DataFrame()
new_frame['AQI'] = data['CO AQI']
new_frame['mean'] = data['CO Mean']
print(len(new_frame))


# In[5]:


frame = new_frame.dropna(how="any")


# In[6]:


len(frame)


# In[7]:


frame.describe()


# In[8]:


x_train , x_test , y_train , y_test = train_test_split(frame[['mean']],frame['AQI'],test_size=0.1,random_state=3)
print(len(x_train))
print(len(y_test))


# In[9]:


model = LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(mean_squared_error(y_test , y_predict))
print(model.coef_)
print(model.intercept_)


# In[10]:


weights  = 17.31961351
bias = -0.1
learning_rate = 0.001
for i in range(0,len(x_train)):
    y_predicted = weights * x_train.values[i] + bias
    error = y_predicted - y_train.values[i]
    weights = weights - learning_rate * error
    bias = bias - learning_rate * error
print(weights)
print(bias)
opt = list()
for i in range(0, len(x_test)):
    final_y = weights * x_test.values[i] + bias
    opt.append(final_y)   
print(len(opt))
print(len(y_predict))    


# In[11]:


redreg = Ridge(alpha=0.1)
redreg.fit(x_train  , y_train)
re_predict = redreg.predict(x_test)
print(mean_squared_error(re_predict , y_test ))

print(redreg.coef_)
print(redreg.intercept_)
    


# In[12]:


lasso = Lasso(alpha=0.1)
lasso.fit(x_train , y_train)
lasso_pre = lasso.predict(x_test)
print(mean_squared_error(y_test , lasso_pre))
print(lasso.coef_)
print(lasso.intercept_)


# In[13]:


elN = ElasticNet(alpha=0.1)
elN.fit(x_train , y_train)
elN_pre = lasso.predict(x_test)
print(mean_squared_error(y_test , elN_pre))
print(elN.coef_)
print(elN.intercept_)


# In[14]:


plt.figure(figsize=(16,7))
plt.scatter(x_test , y_test , label = "actual_data")
plt.scatter(x_test , opt , label = "neural_network")
plt.scatter(x_test , y_predict , label = "linear_regression")
plt.scatter(x_test , re_predict , label  = "ridge regression")
plt.scatter(x_test , lasso_pre , label = "lasso regression")
plt.scatter(x_test , elN_pre , label = "elastic net")
plt.legend()






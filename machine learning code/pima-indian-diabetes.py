#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
column=['Number_of_times_pregnant','glucose_test','blood_pressure','thickness','insulin','Body_mass_index','pedigree_function','Age','class']
data = pd.read_csv("C:/Users/dhruv/Desktop/python sample file/pima-indians-diabetes.csv",names=column)
print(data.head())
print(data.corr())
plt.figure(figsize=(10,7))
plt.scatter(data['glucose_test'],data['class'])
target = data['class']
data.drop(['class'],axis='columns',inplace=True)
x_train , x_test ,y_train,y_test = train_test_split(data,target , test_size=0.1,random_state=4)
print("length of training data",len(x_train))
print("length of testing data",len(x_test))
model = BernoulliNB()
model.fit(x_train , y_train)
y_predicted = model.predict(x_test)
print(mean_squared_error(y_test,y_predicted))
print("accuracy of bernulli",accuracy_score(y_test , y_predicted))
count=0
for i in range(0,len(x_test)):
    if y_predicted[i] == y_test.values[i]:
        count = count+1
print(count)        
print(len(y_test))
tree = RandomForestClassifier(n_estimators=10)
tree.fit(x_train , y_train)
tree_predicted = tree.predict(x_test)
print(mean_squared_error(tree_predicted,y_test))
print("random fotest",accuracy_score(tree_predicted,y_test))
count1=0
for i in range(0,len(x_test)):
    if tree_predicted[i]==y_test.values[i]:
        count1=count1+1
print(count1)


































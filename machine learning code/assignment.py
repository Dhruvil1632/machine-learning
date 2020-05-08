#import necessary libreries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import Lasso
from sklearn.metrics import accuracy_score , mean_squared_error
#load csv file in jupyter notebook
dataset  = pd.read_csv("C:/Users/dhruv/Desktop/dataset/Knight ML Assignment/Data/train.csv")
dataset.keys()
dataset.drop(['user_name','review_title'] , axis = 'columns' , inplace = True)
# data preprocessing and data labeling
list_of_country = []
for i in dataset['country']:
    list_of_country.append(str(i))
l_country = LabelEncoder()
dataset['Country'] = l_country.fit_transform(list_of_country)
province = []
for i in dataset['province']:
    province.append(str(i))
l_province = LabelEncoder()
dataset['Province'] = l_province.fit_transform(province)
region_1 = []
for i in dataset['region_1']:
    region_1.append(str(i))
designation = []
for i in dataset['designation']:
    designation.append(str(i))
l_designation = LabelEncoder()
dataset['Designation'] = l_designation.fit_transform(designation)
l_region_1 = LabelEncoder()
dataset['Region_1'] = l_region_1.fit_transform(region_1)
l_winery = LabelEncoder()
dataset['Winery'] = l_winery.fit_transform(dataset['winery'])
l_variety = LabelEncoder()
dataset['Variety'] = l_variety.fit_transform(dataset['variety'])
region_2 = []
for i in dataset['region_2']:
    region_2.append(str(i))
l_region_2 = LabelEncoder()
dataset['Region_2'] = l_region_2.fit_transform(region_2)
dataset.drop(['country' , 'province' , 'region_1' ,'region_2' ,  'winery','variety','designation'] , axis = 'columns' , inplace = True)
dataset.fillna(0 , inplace = True)
dataset.dropna(how='any' , inplace = True)
# create temporary dataframe while feature extraction
sample = pd.DataFrame()
sample['country'] = dataset['Country']
sample['region_1'] = dataset['Region_1']
sample['region_2'] = dataset['Region_2']
sample['province'] = dataset['Province']
sample['winery'] = dataset['Winery']
sample['designation'] = dataset['Designation']
# spliting of data for training and cross validataion
# x_test and y_test is for cross validation , separate train.csv file work as future unseen data 
x_train , x_test , y_train , y_test = train_test_split(sample , dataset['Variety'] , test_size = 0.1 , random_state = False)
#model creation
regressor = Lasso(alpha = 0.01 , max_iter = 10000)
regressor.fit(x_train , y_train)
prediction = regressor.predict(x_test)
#performance metrics
mse = np.mean(np.sqrt(np.power(prediction - y_test , 2)))
print(mse)
plt.scatter(range(0,len(y_test)), y_test , label = "actual_data")
plt.scatter(range(0,len(prediction)) , prediction , label = "predicted_data")
plt.legend()







 #### 1. Linear Regression
Linear Regression is the process of finding a line that best fits the data points
available on the plot, so that we can use it to predict output values for inputs
that are not present in the data set we have, with the belief that those outputs 
would fall on the line.
In this project I worked on accuracy of linear regression model and OLS model . I shared picture to check accuracy of both models
![](https://github.com/Dhruvil1632/output/blob/master/output%20images/linear%20regression.PNG)

#### 2. Naïve Bayes
Naive Bayes is a classification algorithm for binary (two-class) and multiclass classification problems. It is called Naive Bayes or idiot Bayes because the calculations of the probabilities for each class are simplified to make their calculations tractable
With this method ,  I performed Email is spam or not ?
for better performance I checked every naive bayes algorithm .
![](https://github.com/Dhruvil1632/output/blob/master/output%20images/spam%20detection.PNG)

#### 3.  k-nearest neighbor
k-nearest neighbor algorithm in Python. Supervised Learning : It is the learning where the value or result that we want to predict is within the training data (labeled data) and the value which is in data that we want to study is known as Target or Dependent Variable or Response Variable 
With this algorithm I tried to predicted size of tea shirt using height and weight of the person.
![KNN](https://github.com/Dhruvil1632/output/blob/master/output%20images/KNN.PNG)

#### 4.  Kmean
K-means Clustering in Python. K-means clustering is a clustering algorithm that aims to partition n observations into k clusters. There are 3 steps: Initialisation – K initial “means” (centroids) are generated at random. K clusters are created by associating each observation with the nearest centroid.
With this algorithm I predicted quality of wine using features of wine .
i) below image shows actual data without clustering
![training data](https://github.com/Dhruvil1632/output/blob/master/output%20images/Kmean%20training.PNG)

ii) below image shows clustered data
![](https://github.com/Dhruvil1632/output/blob/master/output%20images/Kmean%20output.PNG)

#### 5.  support vector machine
Support Vector Machine Python Example. Support Vector Machine (SVM) is a supervised machine learning algorithm capable of performing classification, regression and even outlier detection. The linear SVM classifier works by drawing a straight line between two classes. This is where the LSVM algorithm comes in to play.
I used iris dataset . You can check accuracy by below picture.
![SVM](https://github.com/Dhruvil1632/output/blob/master/output%20images/SVM.PNG)

#### 6. random forest 
Random Forest Algorithm with Python and Scikit-Learn. Random forest is a type of supervised machine learning algorithm based on ensemble learning. The random forest algorithm combines multiple algorithm of the same type i.e. multiple decision trees, resulting in a forest of trees,
Random forest is strong learner . accuracy is very important factor for prediction of disease . I took desicion tree as base estimators and implement it. 
![random forest](https://github.com/Dhruvil1632/output/blob/master/output%20images/SVM.PNG)

#### 7. time series analysis
Time series forecasting is the use of a model to predict future values based on previously observed values. Time series are widely used for non-stationary data, like economic, weather, stock price, and retail sales
For prediction of stock of Tata I used LSTMs unit . because it has some additional functionality . It can store information of past . accuracy of  this algorithm you can show by below picture. In this project first i maintain stationarity of data , after prepare dataset for training . for accuracy I used cross-validation method. Output or accuracy you can check by below chart.
![price prediction](https://github.com/Dhruvil1632/output/blob/master/output%20images/price%20prediction.PNG)

#### 8. Convolutional neural network
A specific kind of such a deep neural network is the convolutional network, which is commonly referred to as CNN or ConvNet. It's a deep, feed-forward artificial neural network. Remember that feed-forward neural networks are also called multi-layer perceptron(MLPs), which are the quintessential deep learning models.
I used this algorithm for image classification .In this project first I converted every images in gray scale , after resize  and labeled them , i slicied them in training and testing data . I created classifier as shown in below. after  I trainned model and predict test data.
![image classification](https://github.com/Dhruvil1632/output/blob/master/output%20images/image%20classifier.PNG)



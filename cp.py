# import libraries
# NumPy is a Python library used for working with arrays. 
import numpy as np

# Pandas is a Python library that is used for faster data analysis, data cleaning and data pre-processing
import pandas as pd

# matplotlib used for data visualization.It is a cross-platform library for making 2D plots from data 
import matplotlib.pyplot as plt

# Seaborn library aims to make a more attractive visualization of the central part of understanding and exploring data
import seaborn as sns

#Classification report is used to measure the quality of predictions from a classification algorithm. 
    #How many predictions are True and how many are False. More specifically, True Positives, False Positives, True negatives and 
    #False Negatives are used to predict the metrics of a classification report 

from sklearn.metrics import classification_report
# Sklearn metrics are import metrics in SciKit Learn API to evaluate your machine learning algorithms
from sklearn import metrics

from sklearn import tree

#reading csv file
df = pd.read_csv('Crop_recommendation.csv')

#print the first 5 rows
df.head()

#print the last 5 rows
df.tail()

# to calculate the number of values that are present in the rows and ignore all the null or NaN values
df.count()

# Return an int representing the number of elements in this object.
print(df.size)

# Representing the dimensionality of the DataFrame.
print(df.shape)

# to identify the column name in that index position and pass that name to the drop method
print(df.columns)

#print all unique values in a label column
df['label'].unique()

# to find out the data type of each column 
df.dtypes

# return the count of unique occurences in the specified column.
df['label'].value_counts()

# find the pairwise correlation of all columns in the dataframe.
x = df.corr()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["label"]=le.fit_transform(df["label"])
l=df.dropna()
l

# annot = True is usually passed in the sns.heatmap () function 
#to display the correlation coefficient to facilitate easier interpretation of the heatmap.
sns.heatmap(x,annot=True)
features =df [['N','P','K','temperature','humidity','ph','rainfall']]
target = df['label']
labels = df['label']
acc = []
model = []

# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split the data between the Training Data and Test Data
xtrain,xtest,ytrain,ytest =train_test_split(features,target,test_size=0.2,random_state=2)
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# creating a RF classifier 
RF = RandomForestClassifier(n_estimators=20, random_state=0)

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters

#Train the model using the training sets predicted_values = RF.predict(Xtest)
RF.fit(xtrain,ytrain)

# performing predictions on the test dataset
predicted_values = RF.predict(xtest)

# using metrics module for accuracy calculation
x = metrics.accuracy_score(ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)
print(classification_report(ytest,predicted_values))

#cross_val_predict() function to get the list of values predicted using the model.
#Here we have used 5-fold cross validation(specified by the cv parameter)
from sklearn.model_selection import cross_val_score
score = cross_val_score(RF,features,target,cv=5)
score
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, predicted_values)))
# Import support vector machine Model
from sklearn.svm import SVC

# Building a Support Vector Machine on train data
SVM = SVC(gamma='auto')

#Train the model using the training sets predicted_values = SVM.predict(Xtest)
SVM.fit(xtrain,ytrain)
predicted_values = SVM.predict(xtest)

# using metrics module for accuracy calculation
x = metrics.accuracy_score(ytest, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x)
print(classification_report(ytest,predicted_values))
from sklearn.model_selection import cross_val_score
score = cross_val_score(SVM,features,target,cv=5)
score
# To model the  DecisionTree Classifier
from sklearn.tree import DecisionTreeClassifier

# criterion=entropy -> Criterion is used to measure the quality of split, which is calculated by information gain given by entropy.
# max_depth=5 -> The maximum depth of the tree. 
DecisionTree = DecisionTreeClassifier(criterion='entropy',random_state=2,max_depth=5)

#Train the model using the training sets predicted_values = DecisionTree.predict(Xtest)
DecisionTree.fit(xtrain,ytrain)
predicted_values = DecisionTree.predict(xtest)

# using metrics module for accuracy calculation
x = metrics.accuracy_score(ytest, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("DecisionTrees's Accuracy is: ", x*100)
print(classification_report(ytest,predicted_values))
from sklearn.model_selection import cross_val_score
score = cross_val_score(DecisionTree, features, target,cv=5)
score
# dpi is used to set the resolution of the figure in dots-per-inch.
#figsize() takes two parameters- width and height (in inches). By default the values for width and height are 6.4 and 4.8 respectively. 
plt.figure(figsize=[10,5],dpi = 100)

plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')

#palette can be responsible for generating the different colormap values.
sns.barplot(x = acc,y =model,palette='dark')
data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
prediction = SVM.predict(data)
print(prediction)

data = np.array([[72,41,36,24.098744,80.572268,6.187747,176.860411]])
prediction = NaiveBayes.predict(data)
print(prediction)

data = np.array([[90,42,43,20.879744,82.002744,6.502985,202.935536]])
prediction = RF.predict(data)
print(prediction)

data = np.array([[40,58,75,18.591908,14.779596,7.168096,89.609825]])
prediction = DecisionTree.predict(data)
print(prediction)

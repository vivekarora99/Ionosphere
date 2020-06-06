# -*- coding: utf-8 
"""
Created on Sat Jun  6 14:41:30 2020

@author: Vivek Arora
"""
# Ionosphere_ Decision Tree 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('ionosphere.data')
#checking for missing value
print(dataset.isnull())

#converting dataframe into arrays
#all predictors or independent variables are in X whereas y is dependent variable or target value
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

"""test ratio :30%"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,random_state = 0)

from sklearn.tree import DecisionTreeClassifier
C = []
MAX_DEPTH=[1,2,3,4,5]
for i in range(len(MAX_DEPTH)):
    dec_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH[i],)
    C.append(dec_tree.fit(X_train,y_train))

y_pred = dec_tree.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
    
# Prediction
Prediction= []
for i in range(len(MAX_DEPTH)):
    Prediction.append(C[i].predict(X_test)) 
# Results
from sklearn.metrics import accuracy_score
Accuracy=[]
for i in range(len(MAX_DEPTH)):
   print('\nAccuracy Achieved - {}\n'.format(accuracy_score(y_test,Prediction[i])))
   Accuracy.append(accuracy_score(y_test,Prediction[i]))
   
#Using Seaborn library for Data Visualization
import seaborn as sns
sns.set(style='darkgrid')
sns.lineplot(x= MAX_DEPTH, y = Accuracy)

"""test ratio :40%"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40,random_state = 0)

from sklearn.tree import DecisionTreeClassifier
C = []
MAX_DEPTH=[1,2,3,4,5]
for i in range(len(MAX_DEPTH)):
    dec_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH[i],)
    C.append(dec_tree.fit(X_train,y_train))
    
y_pred = dec_tree.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
    
# Prediction
Prediction= []
for i in range(len(MAX_DEPTH)):
    Prediction.append(C[i].predict(X_test)) 
# Results
from sklearn.metrics import accuracy_score
Accuracy=[]
for i in range(len(MAX_DEPTH)):
   print('\nAccuracy Achieved - {}\n'.format(accuracy_score(y_test,Prediction[i])))
   Accuracy.append(accuracy_score(y_test,Prediction[i]))

#Using Seaborn library for Data Visualization
import seaborn as sns
sns.set(style='darkgrid')
sns.lineplot(x= MAX_DEPTH, y = Accuracy)

"""test ratio :50%"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50,random_state = 0)

from sklearn.tree import DecisionTreeClassifier
C = []
MAX_DEPTH=[1,2,3,4,5]
for i in range(len(MAX_DEPTH)):
    dec_tree = DecisionTreeClassifier(max_depth=MAX_DEPTH[i],)
    C.append(dec_tree.fit(X_train,y_train))
    
y_pred = dec_tree.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
    
# Prediction
Prediction= []
for i in range(len(MAX_DEPTH)):
    Prediction.append(C[i].predict(X_test)) 
# Results
from sklearn.metrics import accuracy_score
Accuracy=[]
for i in range(len(MAX_DEPTH)):
   print('\nAccuracy Achieved - {}\n'.format(accuracy_score(y_test,Prediction[i])))
   Accuracy.append(accuracy_score(y_test,Prediction[i]))

#Using Seaborn library for Data Visualization
import seaborn as sns
sns.set(style='darkgrid')
sns.lineplot(x= MAX_DEPTH, y = Accuracy)
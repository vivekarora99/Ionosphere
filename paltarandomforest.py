import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Importing the dataset
dataset = pd.read_csv('ionosphere.data', header=None)
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

#converting categorical variable to non ategorical variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

n= [8,16,24,32,40]
stat= np.empty((0,4))
''' For given test split and Estimators'''
for i in (30,40,50):
    print("   For test Ratio = "+str(i)+"%" )
    for j in n:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i/100, random_state = 0)


        # Fitting random forest to the Training set
        classifier = RandomForestClassifier(n_estimators = j, criterion = 'entropy', random_state = 5 )
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        # Confusion Matrix and Scores
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test,y_pred)
        acc = accuracy_score(y_test,y_pred)
        print("Test Ratio="+str(i)+"% ,n_estimators="+str(j))
        print('Confusion Matrix:')
        print(cm)
        print('F1 Score: '+str(f1))
        print('Accuracy Score: '+str(acc)+'\n')
        stat=np.append(stat,np.array([[i,j,f1,acc]]), axis=0)
        
n= [8,16,24,32,40]
f_30= stat[:5,2]
f_40= stat[range(5,10),2]
f_50= stat[range(10,15),2]

for i in (30,40,50):
    print("   For test Ratio = "+str(i)+"%" )
    for j in n:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = i/100, random_state = 0)


        # Fitting random forest to the Training set
        classifier = RandomForestClassifier(n_estimators = j, criterion = 'entropy', random_state = 5 )
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        # Confusion Matrix and Scores
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test,y_pred)
        acc = accuracy_score(y_test,y_pred)
        print("Test Ratio="+str(i)+"% ,n_estimators="+str(j))
        print('Confusion Matrix:')
        print(cm)
        print('F1 Score: '+str(f1))
        print('Accuracy Score: '+str(acc)+'\n')
        stat=np.append(stat,np.array([[i,j,f1,acc]]), axis=0)
        
n= [8,16,24,32,40]
f_30= stat[:5,2]
f_40= stat[range(5,10),2]
f_50= stat[range(10,15),2]

acc_30= stat[:5,3]
acc_40= stat[range(5,10),3]
acc_50= stat[range(10,15),3]


    plt.figure(figsize=(15,15))
    plt.plot(n,f_30,color='blue', label="Test Ratio 30%",marker='o')
    plt.plot(n,f_40,color='red', label="Test Ratio 40%", marker= 'o')
    plt.plot(n,f_50,color='green', label="Test Ratio 50%", marker='o')
    plt.title('N Value Vs f1 Score, Random State=5')
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('f1 score')
    
    plt.figure(figsize=(15,15))
    plt.plot(n,acc_30,color='blue', label="Test Ratio 30%",marker='o')
    plt.plot(n,acc_40,color='red', label="Test Ratio 40%", marker= 'o')
    plt.plot(n,acc_50,color='green', label="Test Ratio 50%", marker='o')
    plt.title('N Value Vs Accuracy Score, Random State=5')
    plt.legend()
    plt.xlabel('n')
    plt.ylabel('Accuracy Score')

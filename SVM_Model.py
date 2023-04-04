# -*- coding: utf-8 -*-
"""
Created on Mon May 14 13:29:40 2018

@author: ICHIT-Charmian
"""



##################################################################
############         SVM
##################################################################

from __future__ import division, print_function
import numpy as np
from sklearn import datasets, svm, preprocessing 
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import pandas as pd
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from math import exp, expm1,log
import sklearn
print (sklearn.__version__)
#########      1. Support Vector Classification
#########      1.1 Load the Iris dataset

data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Mean_Var.csv', header=0)
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Final Files- Models\Model2.csv', header=0)

data.shape
data.columns
data.drop(data.columns[[0, 1]], axis=1, inplace=True)
X = data.iloc[:,0:17]
y = data.iloc[:,18]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)



model = svm.SVC(C=1.0, kernel='sigmoid', degree=3, gamma='auto', probability = True) # kernel="linear", probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
len(y_pred)
len(y_test)
correct_classifications = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        correct_classifications += 1
        
accuracy = 100*correct_classifications/len(y_test)

print(accuracy)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('E:\Results\SVM_Sigmoid_ROC')
plt.show()

#C=1.0, kernel='rbf', degree=3, gamma='auto'
#########       1.2 Use Support Vector Machine with different kinds of kernels and evaluate performance

def evaluate_on_test_data(model=None):
    predictions = model.predict(X_test)
    len(predictions)
    len(y_test)
    
    correct_classifications = 0
    for i in range(len(y_test)):
        if predictions[i] == y_test.iloc[i]:
            correct_classifications += 1
    accuracy = 100*correct_classifications/len(y_test) #Accuracy as a percentage
    return accuracy

model = svm.SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)
acc = evaluate_on_test_data(model)

print("{} % accuracy obtained".format(accuracy))
print(predictions)
np.savetxt('a.txt',predictions,fmt='%d')
thefile = open('test.txt', 'w')
for item in predictions:
  predictions.write("%s\n" % item)
  
kernels = ('linear','poly','rbf')
accuracies = []
for index, kernel in enumerate(kernels):
    model = svm.SVC(kernel=kernel)
    model.fit(X_train, y_train)
    acc = evaluate_on_test_data(model)
    accuracies.append(acc)
    print("{} % accuracy obtained with kernel = {}".format(acc, kernel))
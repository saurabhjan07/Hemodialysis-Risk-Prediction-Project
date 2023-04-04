# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:05:34 2018

@author: Saurabh
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
from math import exp, expm1,log

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('E:\HRV Data Analysis\ML Models\LR_Model1.csv', header=0)
print(data.shape)
data['Class'].value_counts()
sns.countplot(x='Class', data = data, palette = 'hls')
plt.savefig('count_plot')
plt.show()
data.groupby('Class').mean()
data.drop(data.columns[[0, 1, 2, 5, 6, 7, 8, 10]], axis=1, inplace=True)
data.columns


#Check the independence between the independent variables
sns.heatmap(data.corr())
plt.savefig('heat_map')
plt.show()

#Split the data into training and test sets
X = data.iloc[:,0:7]
y = data.iloc[:,8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape


 from sklearn.preprocessing import StandardScaler
 sc_x=StandardScaler()
 X_train=sc_x.fit_transform(X_train)
 X_test=sc_x.transform(X_test)


#Fit logistic regression to the training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


#Predicting the test set results and creating confusion matrix
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()






##KNN Classifier#####################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

################################################################
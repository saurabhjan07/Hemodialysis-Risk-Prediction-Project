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
from sklearn.ensemble import RandomForestClassifier

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FFM_LFM_EventDetails_Recoded_Final.csv', header=0)

data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Mean_Var.csv', header=0)

print(data.shape)
data['Class'].value_counts()
sns.countplot(x='Class', data = data, palette = 'hls')
plt.savefig('E:\HRV Data Analysis\ML Models\count_plot_LR2')
plt.show()
data.groupby('Class').mean()
#data.drop(data.columns[[0, 1, 2, 5, 6, 7, 8, 10]], axis=1, inplace=True)
data.columns

data.drop(data.columns[[0, 1]], axis=1, inplace=True)

#Check the independence between the independent variables
sns.heatmap(data.corr())
plt.savefig('heat_map')
plt.show()

###########check column names############
pos=12
colname = data.columns[pos]
print (colname)

#Split the data into training and test sets
X = data.iloc[:,0:17]
y = data.iloc[:,18]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)
 
########################RANDOM FOREST############################################
 clf = RandomForestClassifier(max_depth=2, random_state=0)
 clf.fit(X_train, y_train)
 
 y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


 

###############################################################################
 
 


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

########################CROSS VALIDATION LOGISTIC REGRESSION#################################

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

c, r = y_train.shape
labels = y_train.reshape(c,)

from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator=classifier, X= X_train,y= labels,cv=10)
a1=accuracies.mean() #accuracy
a2=accuracies.std()


roc= cross_val_score(estimator=classifier, X= x_train,y= labels,cv=10,scoring='roc_auc') 
r2=roc.std() #auc
r1=roc.mean()

rmsd=cross_val_score(estimator=classifier, X= x_train,y= labels,cv=10,scoring='neg_mean_squared_error')
rmsd=rmsd**2 #rmsd
rm1=rmsd.mean()
rm2=rmsd.std()

pre=cross_val_score(estimator=classifier, X= x_train,y= labels,cv=10,scoring='precision')
pre1=pre.mean()
pre2=pre.std()

rec=cross_val_score(estimator=classifier, X= x_train,y= labels,cv=10,scoring='recall')
rec1=rec.mean()
rec2=rec.std()

f_measure=cross_val_score(estimator=classifier, X= x_train,y= labels,cv=10,scoring='f1')
f1=f_measure.mean()
f2=f_measure.std()
###############################################################################################################

##KNN Classifier#####################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Mean_Var.csv', header=0)
data.drop(data.columns[[0, 1, 6, 9, 10, 12, 19, 20, 21, 22, 23]], axis=1, inplace=True)
X = data.iloc[:,2:19]
y = data.iloc[:,20]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape


 from sklearn.preprocessing import StandardScaler
 sc_x=StandardScaler()
 X_train=sc_x.fit_transform(X_train)
 X_test=sc_x.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, knn.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, knn.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='KNN Classifier (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
################################################################





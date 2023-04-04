# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:57:59 2018

@author: ICHIT-Charmian
"""
###IMPORT Imp. Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rc("font", size=14)

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

from math import exp, expm1,log
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
#from imblearn import under_sampling, over_sampling
###############           IMPORT Classifiers                ###################

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

################            Import Dataset Model 1                 ###################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Final Files- Models\Model1.csv', header=0)
data.drop(data.columns[[0, 1, 26, 27, 28, 29, 30, 31, 32]], axis=1, inplace=True)
data.drop(data.columns[[2, 11]], axis=1, inplace=True)
data.drop(data.columns[[5,6,7,8,9,13,14,15,16,17]], axis=1, inplace=True)

X_All = data.iloc[:,0:11]
y_All = data.iloc[:,11]


print (colname)
data.shape
data.columns
X_All.columns
X_All.shape



################            Import Dataset Model 2                 ###################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Final Files- Models\Model2.csv', header=0)
data.drop(data.columns[[0,1,40,41,42,43,44,45,46]], axis=1, inplace=True)
data.drop(data.columns[[5,6,7,8,9,13,14,15,16,17, 21,22,23,24,25,29,30,31,32,33 ]], axis=1, inplace=True)
data.shape
data.columns

X_All = data.iloc[:,0:9]
y_All = data.iloc[:,9]

X_All.columns
X_All.shape
y_All


################            Import Dataset Model 3                 ###################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Final Files- Models\Model3.csv', header=0)

data.drop(data.columns[[0,40,41, 42, 43, 44]], axis=1, inplace=True)

data.drop(data.columns[[5,6,7,8,9,13,14,15,16,17,21,22,23,24,25,29,30,31,32,33]], axis=1, inplace=True)

data.shape
data.columns

X_All = data.iloc[:,0:18]
y_All = data.iloc[:,18]


############ ######################################################   
pos=24
colname = data.columns[pos]


data['Class'].value_counts()
sns.countplot(x='Class', data = data, palette = 'hls')
plt.savefig('E:\Results\count_plot_Model1')
plt.show()
#Check the independence between the independent variables
sns.heatmap(data.corr())
plt.savefig('E:\Results\heatmapModel1')
plt.show()

######################################################################



###############  Randon Under Sampling  ######################################
#rus = RandomUnderSampler(return_indices=True)
#X, y, idx_resampled = rus.fit_sample(X_All, y_All)
#X_res_vis = pca.transform(X)

############### Synthetic minority over-sampling (SMOTE) ######################

sm = SMOTE(random_state=42)
X, y = sm.fit_sample(X_All, y_All)


#################  Split the data into training and test sets   ###############

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


###############           Scale the Data          #############################

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)


###################   Clf1 : Logistic Regression   ############################

clf1 = LogisticRegression(random_state=0)
clf1.fit(X_train, y_train)

y_pred1 = clf1.predict(X_test)   # Predicting on Test Set

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred1)

print(confusion_matrix)

print(classification_report(y_test, y_pred1))

         ###########         Plot ROC Curve      ###############
                
logit_roc_auc = roc_auc_score(y_test, y_pred1)
fpr, tpr, thresholds = roc_curve(y_test, clf1.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('E:\Results\LR\ROC_LR_M3')
plt.show()

###################   Clf2 : kNN Classifier        ############################

clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(X_train, y_train)


y_pred2 = clf2.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred2)
print(confusion_matrix)

#print('Accuracy of K-NN classifier on training set: {:.2f}'.format(clf2.score(X_train, y_train)))
#print('Accuracy of K-NN classifier on test set: {:.2f}'.format(clf2.score(X_test, y_test)))

print(classification_report(y_test, y_pred2))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

knn_roc_auc = roc_auc_score(y_test, clf2.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf2.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='KNN Classifier (area = %0.2f)' % knn_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

#plt.savefig('E:\Results\KNN\ROC_KNN_M3')

plt.show()


###################   Clf4 : SVM                   ############################

clf4 = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto', probability = True) # kernel="linear", probability=True)
clf4.fit(X_train, y_train)

y_pred4 = clf4.predict(X_test)

#len(y_pred4)
#len(y_test)
#correct_classifications = 0
#for i in range(len(y_test)):
#    if y_pred4[i] == y_test[i]:
#        correct_classifications += 1
#        
#accuracy = 100*correct_classifications/len(y_test)
#print(accuracy)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred4)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred4))


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, clf4.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf4.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('E:\Results\SVM_Poly_ROC_M2')
plt.show()

###################   Clf3 : Random Forest         ############################

 clf3 = RandomForestClassifier(max_depth=2, random_state=0)
 clf3.fit(X_train, y_train)
 
 y_pred3 = clf3.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred3)
print(confusion_matrix)

print(classification_report(y_test, y_pred3))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
rf_roc_auc = roc_auc_score(y_test, clf3.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf3.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('E:\Results\RF\ROC_RF_M3')
plt.show()


###################   Clf5 : Naive Bayes           ############################
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

clf5 = GaussianNB()
clf5.fit(X_train, y_train)
y_pred5 = clf5.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred5)
print(confusion_matrix)

print(classification_report(y_test, y_pred5))

clf5_1 = MultinomialNB
clf5_1.fit(X_train, y_train)
y_pred5_1 = clf5_1.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred5_1)
print(confusion_matrix)

print(classification_report(y_test, y_pred5_1))



###################   Clf6 : AdaBoost              ############################

from sklearn.ensemble import AdaBoostClassifier


clf6 = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)

clf6.fit(X_train, y_train)

y_pred6 = clf6.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred6)
print(confusion_matrix)

print(classification_report(y_test, y_pred6))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
rf_roc_auc = roc_auc_score(y_test, clf6.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf6.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AdaBoost (area = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('E:\Results\ROC_AdaBoost_M3')
plt.show()

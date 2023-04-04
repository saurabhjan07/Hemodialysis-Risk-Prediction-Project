# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 23:13:59 2018

@author: ICHIT-Charmian
"""

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
from sklearn.metrics import roc_curve, auc

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
#from imblearn import under_sampling, over_sampling

###############           IMPORT Classifiers                ###################

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

#################################     Import Dataset Model 1             #########################################################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Final Files- Models\Model1.csv', header=0)
X_All = data.iloc[:,0:21]
y_All = data.iloc[:,21]

data.shape
data.columns

data.drop(data.columns[[4,5,6,7,8,9,12,13,14,15,16,17]], axis=1, inplace=True)
X_All = data.iloc[:,0:9]
y_All = data.iloc[:,9]

################            Import Dataset Model 2                 ###################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Final Files- Models\Model2.csv', header=0)
X_All = data.iloc[:,0:37]
y_All = data.iloc[:,37]

data.shape
data.columns

data.drop(data.columns[[3,5,6,7,8,9,11, 13,14,15,16,17, 19, 21,22,23,24,25,27,29,30,31,32,33 ]], axis=1, inplace=True)
X_All = data.iloc[:,0:13]
y_All = data.iloc[:,13]

################            Import Dataset Model 3                 ###################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Final Files- Models\Model3.csv', header=0)
X_All = data.iloc[:,0:38]
y_All = data.iloc[:,38]

data.shape
data.columns

data.drop(data.columns[[5,6,7,8,9,13,14,15,16,17,21,22,23,24,25,29,30,31,32,33]], axis=1, inplace=True)
data.drop(data.columns[[3,6,9,12]], axis=1, inplace=True)
X_All = data.iloc[:,0:14]
y_All = data.iloc[:,14]

#################  Split the data into training and test sets   ###############

X_train1, X_test, y_train1, y_test = train_test_split(X_All, y_All, test_size=0.3, random_state=42)

############### Synthetic minority over-sampling (SMOTE) ######################

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train1, y_train1)

###############           Scale the Data          #############################

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

###################   Clf1 : AdaBoost              ############################

clf1 = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
clf1.fit(X_train, y_train)

###################   Clf2 : kNN Classifier        ############################

clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(X_train, y_train)

###################   Clf3 : SVM                   ############################

clf3 = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto', probability = True) # kernel="linear", probability=True)
clf3.fit(X_train, y_train)

###################   Clf4 : Random Forest         ############################

clf4 = RandomForestClassifier(max_depth=2, random_state=0)
clf4.fit(X_train, y_train) 

###################   Clf5 : Logistic Regression   ############################

clf5 = LogisticRegression(random_state=0)
clf5.fit(X_train, y_train)


###############################################################################


colors = ['blue', 'red', 'green', 'magenta', 'cyan' ]
linestyles = ['-','--', '-', '-' , '-.']

all_clf = [clf1, clf5, clf3,  clf4, clf2]
clf_labels = ['AdaBoost', 'Logistic Regression', 'SVM', 'Random Forest', 'KNN']

from matplotlib import pyplot
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    y_pred1 = clf.predict(X_test)
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = roc_auc_score(y_test, y_pred1)
    pyplot.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))
    pyplot.plot([0.0, 1.0], [0.0, 1.0],'k-')
    pyplot.legend(loc='lower right')
    pyplot.xlim([-0.05, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.grid()
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    #pyplot.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Round-2 Review\ROC_Model_R1-1.png', dpi=600)
    #pyplot.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Round-2 Review\ROC_Model_R1-2.png', dpi=600)
    #pyplot.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Round-2 Review\ROC_Model_R1-3.png', dpi=600)
    #pyplot.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Round-2 Review\ROC_Model_R2-1.png', dpi=600)
    #pyplot.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Round-2 Review\ROC_Model_R2-2.png', dpi=600)
    pyplot.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Round-2 Review\ROC_Model_R2-3.png', dpi=600)





###COLORS######
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white
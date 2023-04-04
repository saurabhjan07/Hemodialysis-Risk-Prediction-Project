# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 00:52:27 2018

@author: Saurabh
"""

## Random Sampling of Event Class and then running cross validation

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
from sklearn.preprocessing import StandardScaler
#from imblearn import under_sampling, over_sampling
###############           IMPORT Classifiers                ###################

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

################            Import Dataset Model 1                 ###################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\Model1.csv', header=0) ##All HD Session Data File - Model1
data.drop(data.columns[[0,4,5,8,9]], axis=1, inplace=True)
X_All = data.iloc[:,0:8]
y = data.iloc[:,8]

################            Import Dataset Model 2                ###################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\Model2.csv', header=0) ##All HD Session Data File - Model2
data.drop(data.columns[[0]], axis=1, inplace=True)
X_All = data.iloc[:,0:20]
y = data.iloc[:,20]

################            Import Dataset Model 3                ###################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\Model3.csv', header=0) ##All HD Session Data File - Model3
X_All = data.iloc[:,0:22]
y = data.iloc[:,22]

#################################################################
data.shape
data.columns
#################################################################

################            Import Dataset Model 1 FOR 15-HD Sessions                 ###################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model1_HD15Sessions.csv', header=0) ##15 HD Session Data File
data.drop(data.columns[[0]], axis=1, inplace=True)
X_All = data.iloc[:,0:12]
y = data.iloc[:,12]

################            Import Dataset Model 2 FOR 15-HD Sessions                 ###################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model2_HD15Sessions.csv', header=0) ##15 HD Session Data File
data.drop(data.columns[[0]], axis=1, inplace=True)
X_All = data.iloc[:,0:20]
y = data.iloc[:,20]

################            Import Dataset Model 3 FOR 15-HD Sessions                 ###################
data = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model3_HD15Sessions.csv', header=0) ##15 HD Session Data File
X_All = data.iloc[:,0:21]
y = data.iloc[:,21]


### Data Normalization ###
sc_x=StandardScaler()
X = sc_x.fit_transform(X_All)

clf1 = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
clf2 = KNeighborsClassifier(n_neighbors=3)
clf3 = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto', probability = True)
clf4 = RandomForestClassifier(max_depth=2, random_state=0)
clf5 = LogisticRegression(random_state=0)
sm = SMOTE(random_state=42)

#### Stratified K-Fold Cross Validation ###############################################

kf = StratifiedKFold(n_splits=10, shuffle = True, random_state=45) 

RocAuc_score =[]
Recall_score =[]
Precision_score =[]
Accuracy_score =[]


for train_index,test_index in kf.split(X,y):
    #print('{} of KFold {}'.format(i,kf.n_splits))
    
    X_train1, X_test = X[train_index],X[test_index]
    y_train1, y_test = y[train_index],y[test_index]   
    X_train, y_train = sm.fit_sample(X_train1, y_train1)        
    clf1.fit(X_train, y_train)
    ypred = clf1.predict(X_test)
    RocAuc = roc_auc_score(y_test, ypred)
    Recall = recall_score(y_test, ypred, pos_label=1, average='binary')
    Precision = precision_score(y_test, ypred, pos_label=1, average='binary')
    Accuracy = accuracy_score(y_test, ypred)
    RocAuc_score.append(RocAuc)
    Recall_score.append(Recall)
    Precision_score.append(Precision)
    Accuracy_score.append(Accuracy)

print('\nPrecision: ',Precision_score,'\nMean Precision: ',np.mean(Precision_score))
print('\nRecall: ',Recall_score,'\nMean Recall: ',np.mean(Recall_score))
print('\nAccuracy: ',Accuracy_score,'\nMean Accuracy: ',np.mean(Accuracy_score))
print('\nROC-AUC: ',RocAuc_score,'\nMean ROC-AUC: ',np.mean(RocAuc_score))
    

print('\nPrecision: ', Precision_score, '\nMean Precision: ', np.mean(Precision_score), '\nMean Std.: ', np.std(Precision_score))
print('\nRecall: ', Recall_score, '\nMean Recall: ', np.mean(Recall_score),  '\nMean Std.: ', np.std(Recall_score))
print('\nAccuracy: ',Accuracy_score,'\nMean Accuracy: ',np.mean(Accuracy_score), '\nMean Std.: ', np.std(Accuracy_score))
print('\nROC-AUC: ',RocAuc_score,'\nMean ROC-AUC: ',np.mean(RocAuc_score), '\nMean Std.: ', np.std(RocAuc_score))




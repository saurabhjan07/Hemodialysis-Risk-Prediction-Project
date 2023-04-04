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
from scipy import interp
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
data.drop(data.columns[[0]], axis=1, inplace=True)
X_All = data.iloc[:,0:12]
y = data.iloc[:,12]

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


#################################################################
data.shape
data.columns
#################################################################


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


colors = ['blue', 'red', 'green', 'magenta', 'cyan' ]
linestyles = ['-','--', '-', '-' , '--']

all_clf = [clf1, clf2, clf3,  clf4, clf5]
clf_labels = ['AdaBoost', 'kNN', 'SVM', 'RF', 'LR']


for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
        #i = 0
    for train_index,test_index in kf.split(X,y):
        #print('{} of KFold {}'.format(i,kf.n_splits))
        X_train1, X_test = X[train_index],X[test_index]
        y_train1, y_test = y[train_index],y[test_index]   
        X_train, y_train = sm.fit_sample(X_train1, y_train1)  
        clf.fit(X_train, y_train)
        y_pred1 = clf.predict(X_test)
        probas_ = clf.fit(X_train, y_train).predict_proba(X_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = roc_auc_score(y_test, y_pred1) #auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3) #,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            #i += 1    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs) #auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color=clr, linestyle=ls, label=r'%s (mean auc = %0.2f $\pm$ %0.2f)' % (label, mean_auc, std_auc), lw=2, alpha=.8)
    #pyplot.plot(fpr, tpr, color=clr, linestyle=ls, label='%s (auc = %0.2f)' % (label, roc_auc))
    plt.plot([0.0, 1.0], [0.0, 1.0],'k-')
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2) #, label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()
    #plt.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Final Submission\ROC_All_Model1.png', dpi=600)
    #plt.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Final Submission\ROC_All_Model2.png', dpi=600)
    #plt.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Final Submission\ROC_All_Model3.png', dpi=600)
    plt.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Final Submission\ROC_15HD_Model1.png', dpi=600)
    #plt.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Final Submission\ROC_15HD_Model2.png', dpi=600)
    #plt.savefig('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Results\Final Submission\ROC_15HD_Model3.png', dpi=600)
    

    







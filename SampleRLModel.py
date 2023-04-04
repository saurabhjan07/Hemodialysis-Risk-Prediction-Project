# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:29:47 2017

@author: oracle
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import exp, expm1,log



                    
dataset=pd.read_csv('model21.csv')
x=dataset.iloc[:,[1,2,3,4,5]]


#==============================================================================
# def alert(c):
#   if c['PHQ Score'] > 4.0:
#     return(1)
#   else:
#     return(0)
#==============================================================================

    
#==============================================================================
# def alert(c):
#   if c['Perceived_stress'] >15:
#     return(1)
#   else:
#     return(0)    
#==============================================================================
#x['value'] = x.apply(alert, axis=1)    

x1=x.iloc[:,[0,1,2]].values
y1=x.iloc[:,[4]].values
x2=dataset.iloc[:,[1]]

x_train=x1[0:36,:]
y_train=y1[0:36,:]
x_test=x1[36:45,:]
y_test=y1[36:45,:]


 from sklearn.preprocessing import StandardScaler
 sc_x=StandardScaler()
 x_train=sc_x.fit_transform(x_train)
 x_test=sc_x.transform(x_test)


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

c, r = y_train.shape
labels = y_train.reshape(c,)
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator=classifier, X= x_train,y= labels,cv=10)
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

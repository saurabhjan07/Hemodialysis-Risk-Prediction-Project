# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 23:21:29 2018

@author: ICHIT-Charmian
"""

## PROGRAM to SHORTLIST LAST 15 HD-SESSIONS DATA


import os
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\Model1.csv', header=0)  #Loading Main Data File

for i in range(1, 110):
    df1 = df[df.New_PID == i]
    a = df1['Times'].max()
    b = a-15
    if (b>=0):
        for j in range(1, 16):
            b = b+1
            df2 = df1[df1.Times == b]
            count = df2.shape[0]
            if(count<10):
                print(df2.New_PID, df2.Times)
            df2['New_Times'] = j
            df2 = df2.head(10)
            df2.to_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model1_1.csv', sep='\t', header=False, mode = 'a', index = False)




for i in range(1, 110):
    df1 = df[df.New_PID == i]
    a = df1['Times'].max()
    for j in range(0,a):
        b = j+1
        df2 = df1[df1.Times == b]
        df2['New_Times'] = j
        a = df2.shape[0] #len(df1)
        if(a==10):
            df2.to_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model1.csv', sep='\t', header=False, mode = 'a', index = False)
        elif (a > 10):
            df3 = df2.head(10)
            df3.to_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model1.csv', sep='\t', header=False, mode = 'a', index = False)
            df2.to_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model1_1.csv', sep='\t', header=False, mode = 'a', index = False)
        else:
            df2.to_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model1_1.csv', sep='\t', header=False, mode = 'a', index = False)
               
        
len(df)
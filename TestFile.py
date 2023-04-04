# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:58:26 2018

@author: Saurabh
"""

import os
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

 
                  
for root, dirs, files in os.walk('E:\Folder'):
    for file in files:
        filepath = os.path.join(root, file)
        i = 1
        try:
            dataset = pd.read_csv(filepath , delimiter=",")
            dataset=dataset.dropna()
            dataset=dataset.reset_index()
            splitResult = file.split( "_" )   #split on underscores
            Date = splitResult[2] #first part of the split is the sequence
            Id = splitResult[4]
            df1 = dataset[:10]
            df2 = dataset[10:20]
            df1=df1.reset_index()
            df2=df2.reset_index()
            df=pd.concat([df1, df2], axis=1)
            df['Date'] = Date+' '
            df['Id'] = Id
            df.to_csv('E:\HRV Data Analysis\out.csv', sep='\t', header=False, mode = 'a', index = False)
            #df1.to_csv('E:\HRV Data Analysis\out1.csv',sep='\t', header=False, mode = 'a', index = False)
            #df2.to_csv('E:\HRV Data Analysis\out2.csv',sep='\t', header=False, mode = 'a', index = False)
        #except IndexError:    
        except (IndexError, FileNotFoundError) as e:
            pass

df_ffm = pd.read_csv('E:\HRV Data Analysis\out1.csv', delimiter = ",")
df_lfm = pd.read_csv('E:\HRV Data Analysis\out2.csv', delimiter = ",")

df_ffm.head()

merged_df = pd.merge(left=df_ffm,right=df_lfm, left_on=['Date_FFM', 'P_ID_FFM'] , right_on=['Date_LFM', 'P_ID_LFM'])
merged_df
# what's the size of the output data?
merged_df.shape
merged_df
merged_df.head()
merged_df.to_csv('E:\HRV Data Analysis\out.csv', sep=',')                

df = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\FFM_LFM.csv', delimiter = ",")
df.shape
df=df.dropna()
df.to_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\FFM_LFM_Final.csv', sep='\t', index = False)

df1 = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\FFM_LFM_Final.csv', delimiter = ",")
df1.shape
df1.head()
demography = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\Demographic_Final.csv', delimiter = ",")
demography


merged_df = pd.merge(left=df1,right=demography, left_on='P_ID' , right_on='pid')
merged_df.shape
merged_df.head()


pd.merge(df1.drop('Val', 1), monthly.drop('Date', 1), on=['Year','Month'])


for root, dirs, files in os.walk('E:\Data-Taiwan\Folder'):
    for file in files:
        filepath = os.path.join(root, file)
        i = 1
        try:
            dataset = pd.read_csv(filepath , delimiter=",")
            #dataset=dataset.dropna()
            #dataset=dataset.reset_index()
            splitResult = file.split( "_" )   #split on underscores
            Date = splitResult[1] #first part of the split is the sequence
            Id = splitResult[3]
            #df1 = dataset[:10]
            #df2 = dataset[10:20]
            #df1=df1.reset_index()
            #df2=df2.reset_index()
            #df=pd.concat([df1, df2], axis=1)
            dataset['Date'] = Date+' '
            dataset['Id'] = Id
            dataset.to_csv('E:\HRV Data Analysis\outR.csv', sep='\t', header=False, mode = 'a', index = False)
            #df1.to_csv('E:\HRV Data Analysis\out1.csv',sep='\t', header=False, mode = 'a', index = False)
            #df2.to_csv('E:\HRV Data Analysis\out2.csv',sep='\t', header=False, mode = 'a', index = False)
        #except IndexError:    
        except (IndexError, FileNotFoundError) as e:
            pass

















# -*- coding: utf-8 -*-
"""
Created on Thu May  3 16:14:59 2018

@author: ICHIT-Charmian
"""

import os
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FFM_LFM_EventDetails.csv', delimiter = ",")
df_pid = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\P_ID.csv', delimiter = ",")
df.head()
pid = df_pid['P_ID'].tolist()

j=1

for i in range(0, 109):
    filtered_data = df[df.P_ID == pid[i]] 
    filtered_data['New_PID'] = j
    filtered_data.to_csv('E:\HRV Data Analysis\FFM_LFM_EventDetails_Recoded.csv', sep='\t', header=False, mode = 'a', index = False)
    j = j+1
    
df = pd.read_csv('E:\HRV Data Analysis\FFM_LFM_EventDetails_Recoded.csv', delimiter = ",")
df.shape

for i in range(0, 109):
    filtered_data = df[df.P_ID == pid[i]]
    dates = filtered_data.Date.unique()
    k=1;
    for j in range(0, len(dates)):
        filtered_data_new = filtered_data[filtered_data.Date == dates[j]]
        filtered_data_new['New_Date'] = k
        filtered_data_new.to_csv('E:\HRV Data Analysis\FFM_LFM_EventDetails_Recoded1.csv', sep='\t', header=False, mode = 'a', index = False)
        k=k+1
        

for i in range(0, 109):
    filtered_data = df[df.P_ID == pid[i]]
    dates = filtered_data.Date.unique()
    k=1;
    for j in range(0, len(dates)):
        filtered_data_new = filtered_data[filtered_data.Date == dates[j]]
        filtered_data_new = filtered_data_new.reset_index()    
        filtered_data_new['New_Date'] = k
        #FFM MEANS
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HR_FFM'] = filtered_data_new["HR_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_RR_FFM'] = filtered_data_new["RR_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_MD_FFM'] = filtered_data_new["MD_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HF_FFM'] = filtered_data_new["HF_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_LF_FFM'] = filtered_data_new["LF_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HF/LF_FFM'] = filtered_data_new["HF/LF_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_VLF_FFM'] = filtered_data_new["VLF_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_(VLF+LF)/HF_FFM'] = filtered_data_new["(VLF+LF)/HF_FFM"].mean()
        #LFM MEANS
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HR_LFM'] = filtered_data_new["HR_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_RR_LFM'] = filtered_data_new["RR_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_MD_LFM'] = filtered_data_new["MD_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HF_LFM'] = filtered_data_new["HF_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_LF_LFM'] = filtered_data_new["LF_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HF/LF_LFM'] = filtered_data_new["HF/LF_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_VLF_LFM'] = filtered_data_new["VLF_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_(VLF+LF)/HF_LFM'] = filtered_data_new["(VLF+LF)/HF_LFM"].mean()
        #Write to CSV FILE
        filtered_data_new.to_csv('E:\HRV Data Analysis\FFM_LFM_EventDetails_Recoded12.csv', sep='\t', header=False, mode = 'a', index = False)
        k=k+1


df = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FFM_LFM_EventDetails_Recoded.csv', delimiter = ",")       
df_pid = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\P_ID.csv', delimiter = ",")
pid = df_pid['P_ID'].tolist()
        
for i in range(0, 109):
    filtered_data = df[df.P_ID == pid[i]]
    dates = filtered_data.Date.unique()
    k=1;
    for j in range(0, len(dates)):
        filtered_data_new = filtered_data[filtered_data.Date == dates[j]]
        filtered_data_new = filtered_data_new.reset_index()
        
        Count_Row=filtered_data_new.shape[0] #gives number of row count
        if(Count_Row < 11):
            #Date_List = Date_List.extend(dates[j])
            #print("%d %d" % (filtered_data_new['P_ID'].values[0] ,dates[j] ))
            filtered_data_new['New_Date1'] = k
            filtered_data_new.to_csv('E:\HRV Data Analysis\FFM_LFM_EventDetails_Recoded_Final.csv', sep='\t', header=False, mode = 'a', index = False)
            k=k+1
            
            
            
            
            
            
            
            
        else:
            if(filtered_data_new['Time_LFM'].values[0] < filtered_data_new['Time_LFM'].values[10]):
                filtered_data_new1 = filtered_data_new.iloc[10:20]
                filtered_data_new1.reset_index()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HR_FFM'] = filtered_data_new1["HR_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_RR_FFM'] = filtered_data_new1["RR_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_MD_FFM'] = filtered_data_new1["MD_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HF_FFM'] = filtered_data_new1["HF_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_LF_FFM'] = filtered_data_new1["LF_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HF/LF_FFM'] = filtered_data_new1["HF/LF_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_VLF_FFM'] = filtered_data_new1["VLF_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_(VLF+LF)/HF_FFM'] = filtered_data_new1["(VLF+LF)/HF_FFM"].mean()
                #LFM MEANS
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HR_LFM'] = filtered_data_new1["HR_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_RR_LFM'] = filtered_data_new1["RR_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_MD_LFM'] = filtered_data_new1["MD_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HF_LFM'] = filtered_data_new1["HF_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_LF_LFM'] = filtered_data_new1["LF_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HF/LF_LFM'] = filtered_data_new1["HF/LF_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_VLF_LFM'] = filtered_data_new1["VLF_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_(VLF+LF)/HF_LFM'] = filtered_data_new1["(VLF+LF)/HF_LFM"].mean()
                
                filtered_data_new1.to_csv('E:\HRV Data Analysis\FFM_LFM_EventDetails_Recoded123.csv', sep='\t', header=False, mode = 'a', index = False)
           
            else:
                filtered_data_new1 = filtered_data_new.iloc[0:10]
                filtered_data_new1.reset_index()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HR_FFM'] = filtered_data_new1["HR_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_RR_FFM'] = filtered_data_new1["RR_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_MD_FFM'] = filtered_data_new1["MD_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HF_FFM'] = filtered_data_new1["HF_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_LF_FFM'] = filtered_data_new1["LF_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HF/LF_FFM'] = filtered_data_new1["HF/LF_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_VLF_FFM'] = filtered_data_new1["VLF_FFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_(VLF+LF)/HF_FFM'] = filtered_data_new1["(VLF+LF)/HF_FFM"].mean()
                #LFM MEANS
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HR_LFM'] = filtered_data_new1["HR_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_RR_LFM'] = filtered_data_new1["RR_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_MD_LFM'] = filtered_data_new1["MD_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HF_LFM'] = filtered_data_new1["HF_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_LF_LFM'] = filtered_data_new1["LF_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_HF/LF_LFM'] = filtered_data_new1["HF/LF_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_VLF_LFM'] = filtered_data_new1["VLF_LFM"].mean()
                filtered_data_new1.loc[filtered_data_new1.index[0],'mean_(VLF+LF)/HF_LFM'] = filtered_data_new1["(VLF+LF)/HF_LFM"].mean()
                
                filtered_data_new1.to_csv('E:\HRV Data Analysis\FFM_LFM_EventDetails_Recoded123.csv', sep='\t', header=False, mode = 'a', index = False)
            




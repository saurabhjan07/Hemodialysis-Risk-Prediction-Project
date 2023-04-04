# -*- coding: utf-8 -*-
"""
Created on Wed May 23 12:35:28 2018

@author: ICHIT-Charmian
"""

import os
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




df_pid = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\P_ID.csv', delimiter = ",")
event_df_pid = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Event_P_ID.csv', delimiter = ",")

df = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Mean_Var_All_Parameters.csv', header=0)  #Loading Main Data File
event_df = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\EventDateDetails.csv', delimiter = ",") ##Loading Event date file

dfpid = df_pid['P_ID'].tolist()  ###Getting unique ID's of ALL Patients
eventpid = event_df_pid['P_ID'].tolist()  ###Getting unique ID's of Event Patients

dfpid.sort()
eventpid.sort()

a=0;
for i in range(0, 109):
    
    if(dfpid[i] == eventpid[a]):
        filtered_data1 = df[df.P_ID == dfpid[i]]
        filtered_data1 = filtered_data1.reset_index()
        
        filtered_data2 = event_df[event_df.ID == eventpid[a]]
        filtered_data2 = filtered_data2.reset_index()
        
        dates1 = filtered_data1.Date.unique()
        dates2_1 = filtered_data2.Date.unique()
        dates1.sort()
        dates2_1.sort()
        
        #dates2 = dates1 intersection dates2
        # Python program to illustrate the intersection
        # of two lists in most simple way
        def intersection(dates1, dates2_1):
            list1 = [value for value in dates1 if value in dates2_1]
            return list1
        dates2 = intersection(dates1, dates2_1)
        a=a+1
        b=0
        
        for j in range(0, len(dates1)):
            if(b<len(dates2) and dates1[j] == dates2[b]):
                filtered_data1_1 = filtered_data1[filtered_data1.Date == dates1[j]]
                filtered_data1_1 = filtered_data1_1.reset_index()
                filtered_data1_1.loc[filtered_data1_1.index[0],'ClassEventDate'] = 1
                filtered_data1_1.loc[filtered_data1_1.index[0],'EventDate'] = dates1[j]
                filtered_data1_1.to_csv('E:\HRV Data Analysis\EventDate_Mean_Var_All_Parameters1.csv', sep='\t', header=False, mode = 'a', index = False)
                b=b+1
            else:
                filtered_data1_1 = filtered_data1[filtered_data1.Date == dates1[j]]
                filtered_data1_1 = filtered_data1_1.reset_index()
                filtered_data1_1.loc[filtered_data1_1.index[0],'ClassEventDate'] = 0
                filtered_data1_1.to_csv('E:\HRV Data Analysis\EventDate_Mean_Var_All_Parameters1.csv', sep='\t', header=False, mode = 'a', index = False)          
        
    else:
        filtered_data1 = df[df.P_ID == dfpid[i]]
        filtered_data1 = filtered_data1.reset_index()
        filtered_data1.loc[filtered_data1.index[0],'ClassEventDate'] = 0
        filtered_data1.to_csv('E:\HRV Data Analysis\EventDate_Mean_Var_All_Parameters2.csv', sep='\t', header=False, mode = 'a', index = False)
        
    

   

df = pd.read_csv('E:\HRV Data Analysis\EventDate_Mean_Var_All_Parameters1.csv', header=0)
df_pid = pd.read_csv('E:\HRV Data Analysis\EventDatePID.csv', delimiter = ",")
pid = df_pid['P_ID'].tolist()       
for i in range(0, 61):
    filtered_data = df[df.P_ID == pid[i]]
    filtered_data.reset_index()
    filtered_data.loc[filtered_data.index[0],'AvgSurvivalDays'] = filtered_data["SurvivalDay"].mean()
    filtered_data.to_csv('E:\HRV Data Analysis\EventDate_Mean_Var_All_Parameters1_1.csv', sep='\t', header=False, mode = 'a', index = False)   
    
    
df_pid = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\MasterData.csv', delimiter = ",")










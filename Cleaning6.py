# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:08:52 2018

@author: Saurabh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 14 17:28:34 2018

@author: ICHIT-Charmian
"""

import os
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##### Preparaing File for Model-2 ##########################################

df = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\Final Models_All_Sessions\Final\Model1.csv', header=0)
df = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model1_HD15Sessions.csv', header=0)
pid = (47,69,70,108)

for i in range(1, 110):
    if (i in pid):
        continue
    filtered_data = df[df.New_PID == i]
    a = filtered_data['New_Times'].unique()
    for j in range(0, 15):
        filtered_data_new = filtered_data[filtered_data.New_Times == a[j]]
        filtered_data_new = filtered_data_new.reset_index()
        
        filtered_data_new.loc[filtered_data_new.index[0],'mean_RR_FFM'] = filtered_data_new["RR_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_MD_FFM'] = filtered_data_new["MD_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HR_FFM'] = filtered_data_new["HR_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HF_FFM'] = filtered_data_new["HF_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_LF_FFM'] = filtered_data_new["LF_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HF/LF_FFM'] = filtered_data_new["HF/LF_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_VLF_FFM'] = filtered_data_new["VLF_FFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_(VLF+LF)/HF_FFM'] = filtered_data_new["(VLF+LF)/HF_FFM"].mean()
        
        filtered_data_new.loc[filtered_data_new.index[0],'mean_RR_LFM'] = filtered_data_new["RR_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_MD_LFM'] = filtered_data_new["MD_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HR_LFM'] = filtered_data_new["HR_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HF_LFM'] = filtered_data_new["HF_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_LF_LFM'] = filtered_data_new["LF_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_HF/LF_LFM'] = filtered_data_new["HF/LF_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_VLF_LFM'] = filtered_data_new["VLF_LFM"].mean()
        filtered_data_new.loc[filtered_data_new.index[0],'mean_(VLF+LF)/HF_LFM'] = filtered_data_new["(VLF+LF)/HF_LFM"].mean()
        
        filtered_data_new.loc[filtered_data_new.index[0],'var_RR_FFM'] = filtered_data_new["RR_FFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_MD_FFM'] = filtered_data_new["MD_FFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_HR_FFM'] = filtered_data_new["HR_FFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_HF_FFM'] = filtered_data_new["HF_FFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_LF_FFM'] = filtered_data_new["LF_FFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_HF/LF_FFM'] = filtered_data_new["HF/LF_FFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_VLF_FFM'] = filtered_data_new["VLF_FFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_(VLF+LF)/HF_FFM'] = filtered_data_new["(VLF+LF)/HF_FFM"].var()
        
        filtered_data_new.loc[filtered_data_new.index[0],'var_RR_LFM'] = filtered_data_new["RR_LFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_MD_LFM'] = filtered_data_new["MD_LFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_HR_LFM'] = filtered_data_new["HR_LFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_HF_LFM'] = filtered_data_new["HF_LFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_LF_LFM'] = filtered_data_new["LF_LFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_HF/LF_LFM'] = filtered_data_new["HF/LF_LFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_VLF_LFM'] = filtered_data_new["VLF_LFM"].var()
        filtered_data_new.loc[filtered_data_new.index[0],'var_(VLF+LF)/HF_LFM'] = filtered_data_new["(VLF+LF)/HF_LFM"].var()
#### Write to Final Model 2 ####
        filtered_data_new.to_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model2_1_1.csv', sep='\t', header=False, mode = 'a', index = False)
        


##### Preparaing File for Model-3 ##########################################
df = pd.read_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model2_HD15Sessions.csv', header=0)

    for i in range(1, 110):
        if (i in pid):
            continue
        filtered_data = df[df.New_PID == i]
        filtered_data = filtered_data.reset_index()
        Sessions = filtered_data['New_Times'].tolist()
        k = len(Sessions)
        
        filtered_data.loc[filtered_data.index[0],'Avg_RR_FFM'] = filtered_data["mean_RR_FFM"].mean()
        #filtered_data.loc[filtered_data.index[0],'Avg_MD_FFM'] = filtered_data["mean_MD_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_HR_FFM'] = filtered_data["mean_HR_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_HF_FFM'] = filtered_data["mean_HF_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_LF_FFM'] = filtered_data["mean_LF_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_HF/LF_FFM'] = filtered_data["mean_HF/LF_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_VLF_FFM'] = filtered_data["mean_VLF_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_(VLF+LF)/HF_FFM'] = filtered_data["mean_(VLF+LF)/HF_FFM"].mean()
        
        filtered_data.loc[filtered_data.index[0],'Avg_RR_LFM'] = filtered_data["mean_RR_LFM"].mean()
        #filtered_data.loc[filtered_data.index[0],'Avg_MD_LFM'] = filtered_data["mean_MD_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_HR_LFM'] = filtered_data["mean_HR_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_HF_LFM'] = filtered_data["mean_HF_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_LF_LFM'] = filtered_data["mean_LF_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_HF/LF_LFM'] = filtered_data["mean_HF/LF_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_VLF_LFM'] = filtered_data["mean_VLF_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'Avg_(VLF+LF)/HF_LFM'] = filtered_data["mean_(VLF+LF)/HF_LFM"].mean()
        
        filtered_data.loc[filtered_data.index[0],'AvgVar_RR_FMM'] = filtered_data["var_RR_FFM"].mean()
        #filtered_data.loc[filtered_data.index[0],'AvgVar_MD_FMM'] = filtered_data["var_MD_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_HR_FMM'] = filtered_data["var_HR_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_HF_FFM'] = filtered_data["var_HF_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_LF_FFM'] = filtered_data["var_LF_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_HF/LF_FFM'] = filtered_data["var_HF/LF_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_VLF_FFM'] = filtered_data["var_VLF_FFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_(VLF+LF)/HF_FFM'] = filtered_data["var_(VLF+LF)/HF_FFM"].mean()
        
        filtered_data.loc[filtered_data.index[0],'AvgVar_RR_LFM'] = filtered_data["var_RR_LFM"].mean()
        #filtered_data.loc[filtered_data.index[0],'AvgVar_MD_LFM'] = filtered_data["var_MD_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_HR_LFM'] = filtered_data["var_HR_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_HF_LFM'] = filtered_data["var_HF_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_LF_LFM'] = filtered_data["var_LF_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_HF/LF_LFM'] = filtered_data["var_HF/LF_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_VLF_LFM'] = filtered_data["var_VLF_LFM"].mean()
        filtered_data.loc[filtered_data.index[0],'AvgVar_(VLF+LF)/HF_LFM'] = filtered_data["var_(VLF+LF)/HF_LFM"].mean()
        
        filtered_data.loc[filtered_data.index[0],'Sessions'] = k
        
        filtered_data.to_csv('E:\HRV Data Analysis\HRV_CleanedData_FFM_LFM\MainDataFiles\FinalModels\HDSession_15\Model3_1.csv', sep='\t', header=False, mode = 'a', index = False)
    

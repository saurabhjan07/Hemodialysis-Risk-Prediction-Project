import os, sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import exp, expm1,log


#value_list = [-1,0]
for root, dirs, files in os.walk('E:\HRV Data Analysis\Folder'):
    for file in files:
        filepath = os.path.join(root, file)
        try:
            dataset = pd.read_csv(filepath , delimiter=",")
            dataset.columns = ['Time', 'HR', 'RR', 'MD', 'hrv1', 'hrv2', 'hrv3', 'hrv4', 'hrv5', '']
            dataset=dataset.dropna()
            dataset = dataset[dataset.HR != -1.0]
            dataset = dataset[dataset.HR != 0.0]
            dataset = dataset[dataset.hrv1 != -1.0]
            dataset = dataset[dataset.hrv2 != -1.0]
            dataset = dataset[dataset.hrv4 != -1.0]
            dataset=dataset.reset_index()
            splitResult = file.split( "_" )   #split on underscores
            Date = splitResult[1] #first part of the split is the sequence
            Id = splitResult[3]
            df1 = dataset[:10]
            df2 = dataset[-10:]
            df1=df1.reset_index()
            df2=df2.reset_index()
            df=pd.concat([df1, df2], axis=1)
            df['Date'] = Date+' '
            df['Id'] = Id
            df.to_csv('E:\outRemaining.csv', sep='\t', header=False, mode = 'a', index = False)
            #df1.to_csv('E:\HRV Data Analysis\out1.csv',sep='\t', header=False, mode = 'a', index = False)
            #df2.to_csv('E:\HRV Data Analysis\out2.csv',sep='\t', header=False, mode = 'a', index = False)
        #except IndexError:    
        except (IndexError, FileNotFoundError) as e:
            pass
        
        
        
        
 
dataset = pd.read_csv('E:\s1.csv' , delimiter=",")
dataset.columns = ['Time', 'HR', 'RR', 'MD', 'hrv1', 'hrv2', 'hrv3', 'hrv4', 'hrv5', '']
dataset
dataset=dataset.dropna()
dataset = dataset[dataset.HR != -1.0]
dataset = dataset[dataset.HR != 0.0]
dataset = dataset[dataset.hrv1 != -1.0]
dataset = dataset[dataset.hrv2 != -1.0]
dataset = dataset[dataset.hrv4 != -1.0]
dataset= dataset.replace(to_replace=-1, value='na')
dataset
dataset=dataset.reset_index()
dataset.columns = ['', 'Time', 'HR', 'RR', 'MD', 'HF', 'LF', 'LF/HF', 'VLF', '(VLF+LF)/HF']
df1 = dataset[:10]
df2 = dataset[10:20]
dataset.to_csv('sampleout.csv', sep=',', index = False )                
df1.to_csv('sampleout1.csv', sep=',', index = False )                
df2.to_csv('sampleout2.csv', sep=',', index = False )
dataset        
        
        
        
        
        
        
        
        
        
        
        
        
    
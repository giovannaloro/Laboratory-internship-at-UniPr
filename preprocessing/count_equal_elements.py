import pandas as pd
import os 
import scipy 
import numpy as np 

#uplouading dataset 
os.chdir("..")
datasets=['ML_MED_Dataset_Processed.csv']
#selecting important features 
df1 = pd.read_csv(f'dataset/processed_datasets/ML_MED_Dataset_Processed.csv')
df2 = pd.read_csv(f'dataset/processed_datasets/ML_MED_Dataset_test_Processed.csv')
print(df1)
print(df2)
df3 = pd.concat([df1,df2], axis=0)
print(len(df3)-len(df3.drop_duplicates()))
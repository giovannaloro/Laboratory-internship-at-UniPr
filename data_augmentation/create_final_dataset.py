from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
import numpy as np
import pandas 
import os 

#import dataset 
os.chdir("..")
df_original = pandas.read_csv("dataset/processed_datasets/ML_MED_Dataset_train_Processed_label.csv")
df_smotenc = pandas.read_csv("ML_MED_Dataset_train_Processed_smotenc_best.csv")
for i in range(10):
    #select only 0 classes in original dataset
    df_original = df_original[df_original["tipo_operazione"]==0]
    #sample 300 random row
    df_original = df_original.sample(n=100)
    #concat dfs
    df = pandas.concat([df_smotenc,df_original], axis=0)
    #count classes
    df1 = len(df[df["tipo_operazione"]==0])
    df2 = len(df[df["tipo_operazione"]==1])
    df3 = len(df[df["tipo_operazione"]==2])
    print(f"class_0:{df1},class_1:{df2},class_2:{df3}")
    #save dataset
    df.to_csv(f"optimal_dataset_{i}.csv",index=False)

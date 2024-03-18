import pandas 
import numpy as np
import os
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from ctgan import CTGAN
from ctgan import load_demo


#load dataset 
os.chdir('..')
df = pandas.read_csv('dataset/ML_MED_Dataset_train_preprocessed_full_label.csv')
dataset = df.iloc[:,1:47]

#train ctga
# Names of the columns that are discrete
print(list((dataset.iloc[:,4:47])))
discrete_columns = list((dataset.iloc[:,4:47]))
ctgan = CTGAN(epochs=10)
ctgan.fit(dataset, discrete_columns)

# Create synthetic data
for i in range(2):
    df = ctgan.sample(1000)
    print(df)
    df1 = len(df[df["tipo_operazione"]==0]) 
    df2 = len(df[df["tipo_operazione"]==1])
    df3 = len(df[df["tipo_operazione"]==2])
    df4 = len(df[df["tipo_operazione"]==3])
    df5 = len(df[df["tipo_operazione"]==4])
    print(f"class_0:{df1},class_1:{df2},class_2:{df3},class_3:{df4},class_4:{df5}")
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
import numpy as np
import pandas 
import os 

#import dataset 
os.chdir("..")
df = pandas.read_csv('dataset/ML_MED_Dataset_train_preprocessed_full_label.csv')
X_train = df.iloc[:,1:46]
y_train = df.iloc[:,46:47]
print(X_train)
print(y_train)
#count classes
df1 = len(df[df["tipo_operazione"]==0])
df2 = len(df[df["tipo_operazione"]==1])
df3 = len(df[df["tipo_operazione"]==2])
df4 = len(df[df["tipo_operazione"]==3])
df5 = len(df[df["tipo_operazione"]==4])
print(f"class_0:{df1},class_1:{df2},class_2:{df3},class_3:{df4},class_4:{df5}")
for i in range(10):
    #smote and undersampling of majority class
    over_strategy = {0:288,1:150,2:176,3:100,4:100}
    under_strategy = {0:0,1:0,2:100,3:100,4:100}
    over = SMOTENC(categorical_features=[4, 44] ,sampling_strategy=over_strategy)
    under = RandomUnderSampler(sampling_strategy=under_strategy)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X_train, y_train = pipeline.fit_resample(X_train, y_train)
    df = pandas.concat([X_train, y_train], axis=1, join="inner")
    df.to_csv(f"ML_MED_Dataset_smotenc_{i}.csv", index=False)

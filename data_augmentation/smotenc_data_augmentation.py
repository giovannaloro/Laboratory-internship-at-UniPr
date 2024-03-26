from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
import numpy as np
import pandas 
import os 

#import dataset 
os.chdir("..")
df = pandas.read_csv("dataset/processed_datasets/ML_MED_Dataset_train_Processed_label.csv")
X_train = df.iloc[:,0:29]
y_train = df.iloc[:,29:30]
print(X_train)
print(y_train)
#count classes
df1 = len(df[df["tipo_operazione"]==0])
df2 = len(df[df["tipo_operazione"]==1])
df3 = len(df[df["tipo_operazione"]==2])
print(f"class_0:{df1},class_1:{df2},class_2:{df3}")
#set smotenc generation strategy
over_strategy = {0:755,1:300,2:300}
under_strategy = {0:400,1:300,2:300}
categorical_features = list(range(4, 29))
for i in [10,21]:
    #smote and undersampling of majority class
    over = SMOTENC(categorical_features=categorical_features ,sampling_strategy=over_strategy, k_neighbors=9)
    under = RandomUnderSampler(sampling_strategy=under_strategy)
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X_train, y_train = pipeline.fit_resample(X_train, y_train)
    df = pandas.concat([X_train, y_train], axis=1, join="inner")
    df.to_csv(f"ML_MED_Dataset_train_Processed_smotenc_{i}.csv", index=False)

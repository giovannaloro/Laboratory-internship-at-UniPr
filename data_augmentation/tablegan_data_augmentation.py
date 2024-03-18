from tabgan.sampler import OriginalGenerator, GANGenerator, ForestDiffusionGenerator
import pandas 
import numpy as np
import os
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


#uplouading dataset 
os.chdir('..')
df = pandas.read_csv('augmented_dataset/ML_MED_Dataset_smotenc_best.csv')
X_train = df.iloc[:,0:45]
y_train = df.iloc[:,45:46]
print(X_train)
print(y_train)
df = pandas.read_csv('dataset/ML_MED_Dataset_validation_preprocessed_full_label.csv')
X_validation = df.iloc[:,1:46]
y_validation = df.iloc[:,46:47]


for i in range(10):
    # generate data
    X_gen, y_gen = GANGenerator(cat_cols = list((X_train.iloc[:,4:45]
)) , gen_params = {"batch_size": 30, "patience": 25, "epochs" : 500,}, adversarial_model_params={"random_state": np.random.randint(0,1000)}).generate_data_pipe(X_train, y_train, X_validation )
    df = pandas.concat([X_gen, y_gen], axis=1, join="inner")
    df1 = len(df[df["tipo_operazione"]==0])
    df2 = len(df[df["tipo_operazione"]==1])
    df3 = len(df[df["tipo_operazione"]==2])
    df4 = len(df[df["tipo_operazione"]==3])
    df5 = len(df[df["tipo_operazione"]==4])
    print(f"class_0:{df1},class_1:{df2},class_2:{df3},class_3:{df4},class_4:{df5}")
    df.to_csv(f"ML_MED_Dataset_cgansmote_{i}.csv", index=False)



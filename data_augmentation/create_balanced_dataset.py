import pandas 
import os

os.chdir("..")
df = pandas.read_csv("augmented_dataset/ML_MED_Dataset_smotenc_best.csv")
df = df.loc[df['tipo_operazione'] >= 2]
original_df = pandas.read_csv("dataset/ML_MED_Dataset_train_preprocessed_full_label.csv")
df = pandas.concat([df, original_df], axis=0)
df = df.drop(['Unnamed: 0'], axis=1)
df.to_csv("balanced_dataset_train_label.csv", index=False)
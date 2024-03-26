from matplotlib import pyplot
import numpy as np
import pandas 
import os 
from sklearn.model_selection import train_test_split

#import dataset 
os.chdir("..")
df = pandas.read_csv('dataset/processed_datasets/ML_MED_Dataset_Processed_onehot.csv')
X = df.iloc[:,0:29]
y = df.iloc[:,29:32]
#count classes
df1 = len(df[df["INTERVENTI SULL’APPARATO DIGERENTE"]==1])
df2 = len(df[df["INTERVENTI SUL SISTEMA ENDOCRINO"]==1])
df3 = len(df[df["INTERVENTI SUL SISTEMA CARDIOVASCOLARE"]==1])
print(f"class_0:{df1},class_1:{df2},class_2:{df3}")
#train test split
X_train, X_validation, y_train, y_validation  = train_test_split(X, y, test_size=0.2, random_state=1000)
df1 = len(y_train[y_train["INTERVENTI SULL’APPARATO DIGERENTE"]==1])
df2 = len(y_train[y_train["INTERVENTI SUL SISTEMA ENDOCRINO"]==1])
df3 = len(y_train[y_train["INTERVENTI SUL SISTEMA CARDIOVASCOLARE"]==1])
print(f"class_0:{df1},class_1:{df2},class_2:{df3}")
df1 = len(y_validation[y_validation["INTERVENTI SULL’APPARATO DIGERENTE"]==1])
df2 = len(y_validation[y_validation["INTERVENTI SUL SISTEMA ENDOCRINO"]==1])
df3 = len(y_validation[y_validation["INTERVENTI SUL SISTEMA CARDIOVASCOLARE"]==1])
print(f"class_0:{df1},class_1:{df2},class_2:{df3}")
df_train = pandas.concat([X_train, y_train], axis = 1)
df_validation = pandas.concat([X_validation, y_validation], axis = 1)
df_train.to_csv("ML_MED_Dataset_train_Processed_onehot.csv", index=False)
df_validation.to_csv("ML_MED_Dataset_validation_Processed_onehot.csv", index=False)



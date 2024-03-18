import pandas as pd
import os 
import scipy 
import numpy as np 
from joblib import dump, load
from sklearn import datasets
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

#uplouading dataset 
os.chdir("..")
datasets=['MLMED_Dataset_preprocessed_full.csv','ML_MED_Dataset_validation_preprocessed_full.csv','ML_MED_Dataset_test_preprocessed_full.csv','ML_MED_Dataset_train_preprocessed_full.csv']
for dataset in datasets:
    df = pd.read_csv(f'dataset/{dataset}')
    X_num = df.iloc[:,1:5]
    print(X_num)
    X_cat = df.loc[:,['Cardiopatia ischemica cronica','Insulina','ARITMIA_TPSV','ROBOTICA','ENDOSCOPIA','LAPAROSCOPIA','OPEN','ASA_2.0','Antipertensivi','Fumo','MALLAMPATI_2.0','MALLAMPATI_4.0','Anticoagulanti','Catetere vescicale','DIABETE_MELLITO_1','TIGO','Antiaritmici']]
    #selecting important features 
    X_num = pd.DataFrame(X_num, columns = ['Età','Peso','Altezza','BMI'])
    X = X_num.join(X_cat)
    y = df.iloc[:,47:52]
    output_dataset = X.join(y)
    #saving preprocessed dataset
    file_name = dataset.split('_full.')[0] + '_important_features' + '.csv'
    output_dataset.to_csv(file_name)

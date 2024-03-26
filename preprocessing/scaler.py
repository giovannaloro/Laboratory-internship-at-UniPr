import pandas as pd
import os 
import scipy 
import numpy as np 
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

#uplouading dataset 
datasets=['ML_MED_Dataset_train_Processed_onehot.csv','ML_MED_Dataset_validation_Processed_onehot.csv','ML_MED_Dataset_test_Processed_onehot.csv','ML_MED_Dataset_KFold_Processed_onehot.csv']
os.chdir("..")
for dataset in datasets:
    df = pd.read_csv(f'dataset/processed_datasets/{dataset}')
    y = df.iloc[:,29:32]
    X_num = df.loc[:,['Età','Peso','Altezza','BMI']]
    X_cat = df.loc[:,['Sesso','Fumo','OSAS','BPCO','Ipertensione arteriosa','Cardiopatia ischemica cronica','Pregresso infarto miocardio','Pregresso SCC','Ictus','Pregresso TIA',	'Altro_comorbidita','Antipertensivi','Broncodilatatori','Antiaritmici','Anticoagulanti','Antiaggreganti','TIGO','Insulina','Altro_terapia','DIABETE_MELLITO_2','DIABETE_MELLITO_1','ARITMIA_NO','ARITMIA_FA','ARITMIA_TACHI','ARITMIA_TPSV']]
    #scaling and selecting dataset
    mms = MinMaxScaler()
    X_num = mms.fit_transform(X_num)
    X_num = pd.DataFrame(X_num, columns = ['Età','Peso','Altezza','BMI'])
    X = X_num.join(X_cat)
    output_dataset = X.join(y)
    #saving preprocessed dataset
    file_name = dataset.split('.')[0] + 'scaled' + '.csv'
    output_dataset.to_csv(file_name, index=False)



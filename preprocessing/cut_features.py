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
datasets=['ML_MED_Dataset_Processed.csv']
for dataset in datasets:
    #selecting important features 
    df = pd.read_csv(f'dataset/processed_datasets/{dataset}')
    X = df.loc[:,['Età','Sesso','Peso','Altezza','BMI','Fumo','OSAS','BPCO','Ipertensione arteriosa','Cardiopatia ischemica cronica','Pregresso infarto miocardio','Pregresso SCC','Ictus','Pregresso TIA',	'Altro_comorbidita','Antipertensivi','Broncodilatatori','Antiaritmici','Anticoagulanti','Antiaggreganti','TIGO','Insulina','Altro_terapia','DIABETE_MELLITO_2','DIABETE_MELLITO_1','ARITMIA_NO','ARITMIA_FA','ARITMIA_TACHI','ARITMIA_TPSV','INTERVENTI SULL’APPARATO DIGERENTE','INTERVENTI SUL SISTEMA ENDOCRINO','INTERVENTI SUL SISTEMA CARDIOVASCOLARE'
]]
    #saving preprocessed dataset
    file_name = dataset.split('_full.')[0] + '_important_features' + '.csv'
    X.to_csv(file_name, index=False)

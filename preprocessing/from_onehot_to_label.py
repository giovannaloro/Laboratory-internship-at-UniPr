from xml.etree.ElementTree import tostring
import pandas
import os 
import scipy 
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

def to_label(dataset_name):
    df = pandas.read_csv(f'dataset/processed_datasets/{dataset_name}')
    X = df.iloc[:,0:29]
    y_onehot = df.iloc[:,29:32]
    print(X)
    print(y_onehot)
    y_label = pandas.DataFrame(columns = ['tipo_operazione'])
    for index, row in y_onehot.iterrows():
        operation = (0*row['INTERVENTI SULL’APPARATO DIGERENTE'] + 1*row['INTERVENTI SUL SISTEMA ENDOCRINO'] +
                    2*row['INTERVENTI SUL SISTEMA CARDIOVASCOLARE'])   
        match operation:
            case 0:
                list_row = [0]
                y_label.loc[len(y_label)] = list_row
            case 1:
                list_row = [1]
                y_label.loc[len(y_label)] = list_row    
            case 2:
                list_row = [2]
                y_label.loc[len(y_label)] = list_row
    output_dataset=X.join(y_label)
    file_name = dataset_name.split('onehot')[0] + 'label' + '.csv'
    output_dataset.to_csv(file_name, index=False)
os.chdir("..")
files=['ML_MED_Dataset_Processed_onehot.csv']
for file in files:
    to_label(file)

from xml.etree.ElementTree import tostring
import pandas
import os 
import scipy 
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

def to_label(dataset_name):
    df = pandas.read_csv(f'dataset/{dataset_name}')
    X = df.iloc[:,1:46]
    y_onehot = df.iloc[:,46:52]
    y_label = pandas.DataFrame(columns = ['tipo_operazione'])
    for index, row in y_onehot.iterrows():
        operation = (1*row['INTERVENTI SUL SISTEMA ENDOCRINO']+0*row['INTERVENTI SULL’APPARATO DIGERENTE'] +
                    2*row['INTERVENTI SULL’APPARATO URINARIO'] + 3*row['INTERVENTI SUL SISTEMA RESPIRATORIO']+
                    4*row['INTERVENTI SUL SISTEMA CARDIOVASCOLARE'])   
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
            case 3:
                list_row = [3]
                y_label.loc[len(y_label)] = list_row
            case 4:
                list_row = [4]
                y_label.loc[len(y_label)] = list_row
    output_dataset=X.join(y_label)
    file_name = dataset_name.split('.')[0] + '_label' + '.csv'
    output_dataset.to_csv(file_name)
os.chdir("..")
files=['MLMED_Dataset_preprocessed_full.csv','ML_MED_Dataset_test_preprocessed_full.csv','ML_MED_Dataset_KFold_preprocessed_full.csv',
       'ML_MED_Dataset_train_preprocessed_full.csv','ML_MED_Dataset_validation_preprocessed_full.csv']
for file in files:
    to_label(file)

import pandas
import os


def to_onehot(dataset_name):
    df = pandas.read_csv(f'augmented_dataset/3classes/{dataset_name}')
    X = df.iloc[:,0:29]
    y_label = df.iloc[:,29:30]
    print(y_label)
    y_onehot = pandas.DataFrame(columns = ['INTERVENTI SUL SISTEMA DIGERENTE','INTERVENTI SULL’APPARATO ENDOCRINO','INTERVENTI SUL SISTEMA CARDIOVASCOLARE'])
    print(y_onehot)
    for index, row in y_label.iterrows():
        operation = row['tipo_operazione']
        match operation:
            case 0:
                list_row = [1,0,0]
                y_onehot.loc[len(y_onehot)] = list_row
            case 1:
                list_row = [0,1,0]
                y_onehot.loc[len(y_onehot)] = list_row    
            case 2:
                list_row = [0,0,1]
                y_onehot.loc[len(y_onehot)] = list_row
        print(y_onehot)
    output_dataset=X.join(y_onehot)
    file_name = dataset_name.split('.')[0] + '_onehot' + '.csv'
    output_dataset.to_csv(file_name, index=False)

files=['optimal_dataset.csv']
os.chdir("..")
for file in files:
    to_onehot(file)

import pandas


def to_onehot(dataset_name):
    df = pandas.read_csv(f'{dataset_name}')
    X = df.iloc[:,0:45]
    y_label = df.iloc[:,45:46]
    print(y_label)
    y_onehot = pandas.DataFrame(columns = ['INTERVENTI SUL SISTEMA DIGERENTE','INTERVENTI SULL’APPARATO ENDOCRINO','INTERVENTI SULL’APPARATO URINARIO','INTERVENTI SUL SISTEMA RESPIRATORIO','INTERVENTI SUL SISTEMA CARDIOVASCOLARE'])
    print(y_onehot)
    for index, row in y_label.iterrows():
        operation = row['tipo_operazione']
        match operation:
            case 0:
                list_row = [1,0,0,0,0]
                y_onehot.loc[len(y_onehot)] = list_row
            case 1:
                list_row = [0,1,0,0,0]
                y_onehot.loc[len(y_onehot)] = list_row    
            case 2:
                list_row = [0,0,1,0,0]
                y_onehot.loc[len(y_onehot)] = list_row
            case 3:
                list_row = [0,0,0,1,0]
                y_onehot.loc[len(y_onehot)] = list_row
            case 4:
                list_row = [0,0,0,0,1]
                y_onehot.loc[len(y_onehot)] = list_row
        print(y_onehot)
    output_dataset=X.join(y_onehot)
    file_name = dataset_name.split('.')[0] + '_onehot' + '.csv'
    output_dataset.to_csv(file_name, index=False)

files=['dataset_ctgan_smotenc_generated_best.csv', 'dataset_rtvae_smotenc_generated_best.csv','ML_MED_Dataset_smotenc_best.csv']
for file in files:
    to_onehot(file)

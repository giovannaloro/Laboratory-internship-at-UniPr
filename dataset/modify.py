import pandas as pd

datasets = ["ML_MED_Dataset_test_preprocessed_full.csv","ML_MED_Dataset_train_preprocessed_full.csv", "ML_MED_Dataset_validation_preprocessed_full.csv"]
for dataset in datasets:
    df = pd.read_csv(dataset)
    df = df.drop(["Unnamed: 0.1"], axis=1)
    df.to_csv(dataset, index=False)
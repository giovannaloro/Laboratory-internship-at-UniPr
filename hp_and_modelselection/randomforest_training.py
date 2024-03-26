from numpy import mean
from numpy import std
import numpy as np
import pandas 
import os 
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,accuracy_score
from sklearn.ensemble import RandomForestClassifier

#uplouading dataset 
os.chdir('..')
models=['optimal']
for model in models:
    df = pandas.read_csv(f'augmented_dataset/3classes/optimal_dataset.csv')
    X_train = df.iloc[:,0:29]
    y_train = df.iloc[:,29:30]
    y_train = np.ravel(y_train)
    df = pandas.read_csv('dataset/processed_datasets/ML_MED_Dataset_validation_Processed_label.csv')
    X_validation = df.iloc[:,0:29]
    y_validation = df.iloc[:,29:30]
    y_validation = np.ravel(y_validation)
    #train 10 models and save 
    best_params = {'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 350, 'n_jobs': -1}

    for i in range(50):
        rf = RandomForestClassifier(**best_params)
        rf.fit(X_train, y_train)
        y_pred =rf.predict(X_validation)
        print(rf.n_classes_)
        f_one_score = f1_score(y_validation, y_pred, average="macro")
        accuracy = accuracy_score(y_validation, y_pred)
        print(f"f1 score with {model} of model {i}:", f_one_score)
        filename = f"rf_{model}_model{i}.sav"
        joblib.dump(rf, filename)



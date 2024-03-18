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
df = pandas.read_csv('augmented_dataset/balanced_dataset_train_label.csv')
X_train = df.iloc[:,0:45]
y_train = df.iloc[:,45:46]
print(X_train)
print(y_train)
y_train = np.ravel(y_train)
df = pandas.read_csv('dataset/ML_MED_Dataset_validation_preprocessed_full_label.csv')
X_validation = df.iloc[:,1:46]
y_validation = df.iloc[:,46:47]
y_validation = np.ravel(y_validation)
#train 10 models and save 
best_params = {'criterion': 'gini', 'max_features': 'sqrt', 'n_estimators': 300, 'n_jobs': -1}


rf = RandomForestClassifier(**best_params)
for i in range(100):
    rf = RandomForestClassifier(**best_params)
    rf.fit(X_train, y_train)
    y_pred =rf.predict(X_validation)
    print(rf.n_classes_)
    f_one_score = f1_score(y_validation, y_pred, average="macro")
    accuracy = accuracy_score(y_validation, y_pred)
    print(f"f1 score of model {i}:", f_one_score)
    filename = f"rf_model{i}.sav"
    joblib.dump(rf, filename)



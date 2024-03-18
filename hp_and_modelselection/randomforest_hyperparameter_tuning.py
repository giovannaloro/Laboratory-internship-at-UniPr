import pandas 
import os 
import scipy
import numpy as np 
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score
#uplouading dataset 
os.chdir('..')
df = pandas.read_csv('dataset/ML_MED_Dataset_train_preprocessed_full.csv')
X_train = df.iloc[:,1:46]
y_train = df.iloc[:,46:52]
df = pandas.read_csv('dataset/ML_MED_Dataset_validation_preprocessed_full.csv')
X_validation = df.iloc[:,1:46]
y_validation = df.iloc[:,46:52]
#training and hyperparamater tuning _____
rf = RandomForestClassifier()
params = {'n_estimators':[150,200,250,300,350,400,450,500], 'criterion':['gini','entropy','log_loss'],'max_features':['sqrt','log2'],
                       'n_jobs':[-1]}
rf_tuned = GridSearchCV(estimator=rf, param_grid=params, n_jobs=-1, cv=8, verbose=3)
rf_tuned.fit(X_train,y_train)
print(rf_tuned.best_params_)
print(rf_tuned.best_score_)
#evaluation
rf = RandomForestClassifier(**rf_tuned.best_params_)
rf.fit(X_train, y_train)
y_pred =rf.predict(X_validation)
f_one_score = f1_score(y_validation, y_pred, average="weighted")
accuracy = accuracy_score(y_validation, y_pred)
print("f1 score:", f_one_score)
print("accuracy score:", accuracy)

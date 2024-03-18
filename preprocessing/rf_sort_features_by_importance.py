import pandas 
import numpy as np 
import joblib
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import os

#uplouading dataset 
os.chdir('..')
df = pandas.read_csv('dataset/ML_MED_Dataset_train_preprocessed_full.csv')
X_train = df.iloc[:,1:47]
y_train = df.iloc[:,47:52]
print(X_train)
#feature selection with importances attribute of random forest
#rf = joblib.load('models/best_randomforest_model.sav') 
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = df.columns[:]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]],importances[indices[f]]))
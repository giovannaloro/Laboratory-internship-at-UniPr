# xgboost for feature importance on a classification problem
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from matplotlib import pyplot
import os
import numpy as np
import pandas

#uplouading dataset 
os.chdir('..')
df = pandas.read_csv('dataset/ML_MED_Dataset_train_preprocessed_full.csv')
X_train = df.iloc[:,1:47]
y_train = df.iloc[:,47:52]
print(X_train)
# define the model
model = XGBClassifier()
# fit the model
model.fit(X_train, y_train)
# get importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = df.columns[:]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]],importances[indices[f]]))
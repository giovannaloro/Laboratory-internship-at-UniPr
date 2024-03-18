import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import joblib
import os 
import matplotlib.pyplot as plt
from sklearn.ensemble import  AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, KFold
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score
from to_image import to_image

#load dataset 
os.chdir("..")
df = pd.read_csv("dataset/ML_MED_Dataset_KFold_preprocessed_full.csv")
X_kf = df.iloc[:,1:46]
X_kf_cnn  = df.iloc[:,1:46]
y_kf = df.iloc[:,46:52]
X_kf_cnn["empty1"] = 0 
X_kf_cnn["empty2"] = 0
X_kf_cnn["empty3"] = 0
X_kf_cnn["empty4"] = 0
X_test_cnn = to_image(X_kf_cnn,7,7,1 )
df = pd.read_csv("dataset/ML_MED_Dataset_KFold_preprocessed_full_label.csv")
y_kf_label = df.iloc[:,46:47]
y_kf_label = np.ravel(y_kf_label)
#randomforest metrics calculation
best_params = {'criterion': 'entropy', 'max_features': 'sqrt', 'n_estimators': 400, 'n_jobs': -1}
rf = RandomForestClassifier(**best_params)
cv = KFold(n_splits=8, random_state=1, shuffle=True)
scores_accuracy = cross_val_score(rf, X_kf, y_kf, scoring='accuracy', cv=cv, n_jobs=-1)
scores_f1micro = cross_val_score(rf, X_kf, y_kf, scoring='f1_micro', cv=cv, n_jobs=-1)
scores_f1macro = cross_val_score(rf, X_kf, y_kf, scoring='f1_macro', cv=cv, n_jobs=-1)
print(f"rf matrics: f1micro:{np.mean(scores_f1micro)}, f1macro:{np.mean(scores_f1macro)}, accuracy:{np.mean(scores_accuracy)}")
#adaboost metrics calculation
best_params = {'algorithm': 'SAMME', 'estimator': SGDClassifier(), 'learning_rate': 0.01, 'n_estimators': 200}
ab = AdaBoostClassifier(**best_params)
cv = KFold(n_splits=8, random_state=1, shuffle=True)
scores_accuracy = cross_val_score(ab, X_kf, y_kf_label, scoring='accuracy', cv=cv, n_jobs=-1)
scores_f1micro = cross_val_score(ab, X_kf, y_kf_label, scoring='f1_micro', cv=cv, n_jobs=-1)
scores_f1macro = cross_val_score(ab, X_kf, y_kf_label, scoring='f1_macro', cv=cv, n_jobs=-1)
print(f"ab matrics: f1micro:{np.mean(scores_f1micro)}, f1macro:{np.mean(scores_f1macro)}, accuracy:{np.mean(scores_accuracy)}")
"""

#cnn metrics calculation
cnn = load_model("models/cnn_best_model.h5")
metrics = cnn.evaluate(X_test_cnn, y_)
print(f"cnn matrics: f1micro:{metrics[3]}, f1macro:{metrics[2]}, accuracy:{metrics[1]}")
#nn metrics calculation
nn = load_model("models/nn_best_model.h5")
metrics = nn.evaluate(X_test, y_test)
print(f"nn matrics: f1micro:{metrics[3]}, f1macro:{metrics[2]}, accuracy:{metrics[1]}")"
"""
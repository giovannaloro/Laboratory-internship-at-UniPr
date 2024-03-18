import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import joblib
import os 
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score

#load dataset 
os.chdir("..")
df = pd.read_csv("dataset/ML_MED_Dataset_test_preprocessed_full.csv")
X_test = df.iloc[:,1:46]
y_test = df.iloc[:,46:52]
df = pd.read_csv("dataset/ML_MED_Dataset_test_preprocessed_full_label.csv")
y_test_label = df.iloc[:,46:47]
y_test_label = np.ravel(y_test_label)
#randomforest metrics calculation
rf = joblib.load("models/rf_model_smotenc_best.sav")
y_pred = rf.predict(X_test)
f_one_micro = f1_score(y_test_label, y_pred, average="micro")
f_one_macro = f1_score(y_test_label, y_pred, average="macro")
accuracy = accuracy_score(y_test_label, y_pred)
print(f"rf matrics: f1micro:{f_one_micro}, f1macro:{f_one_macro}, accuracy:{accuracy}")
#adaboost metrics calculation
ab = joblib.load("models/ab_model_smotenc_best.sav")
y_pred = ab.predict(X_test)
f_one_micro = f1_score(y_test_label, y_pred, average="micro")
f_one_macro = f1_score(y_test_label, y_pred, average="macro")
accuracy = accuracy_score(y_test_label, y_pred)
print(f"ab matrics: f1micro:{f_one_micro}, f1macro:{f_one_macro}, accuracy:{accuracy}")
#nn metrics calculation
nn = load_model("models/trained_model_smotenc_best.h5")
metrics = nn.evaluate(X_test, y_test)
print(f"nn matrics: f1micro:{metrics[3]}, f1macro:{metrics[2]}, accuracy:{metrics[1]}")

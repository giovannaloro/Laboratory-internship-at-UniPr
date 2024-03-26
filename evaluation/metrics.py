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
df = pd.read_csv("dataset/processed_datasets/ML_MED_Dataset_test_Processed_onehot.csv")
X_test = df.iloc[:,0:29]
y_test = df.iloc[:,29:32]
df = pd.read_csv("dataset/processed_datasets/ML_MED_Dataset_test_Processed_label.csv")
y_test_label = df.iloc[:,29:30]
y_test_label = np.ravel(y_test_label)

models = [""]
for model in models:
    #randomforest metrics calculation
    rf = joblib.load(f"models/3classes/rf_{model}_model_best.sav")
    y_pred = rf.predict(X_test)
    f_one_micro = f1_score(y_test_label, y_pred, average="micro")
    f_one_macro = f1_score(y_test_label, y_pred, average="macro")
    accuracy = accuracy_score(y_test_label, y_pred)
    print(f"rf {model} dataset metrics: f1micro:{f_one_micro}, f1macro:{f_one_macro}, accuracy:{accuracy}")
    #adaboost metrics calculation
    ab = joblib.load(f"models/3classes/ab_{model}_model_best.sav") 
    y_pred = ab.predict(X_test)
    f_one_micro = f1_score(y_test_label, y_pred, average="micro")
    f_one_macro = f1_score(y_test_label, y_pred, average="macro")
    accuracy = accuracy_score(y_test_label, y_pred)
    print(f"ab {model} dataset metrics: f1micro:{f_one_micro}, f1macro:{f_one_macro}, accuracy:{accuracy}")
    #nn metrics calculation
    nn = load_model(f"models/3classes/trained_model_{model}_best.h5")
    metrics = nn.evaluate(X_test, y_test)
    print(f"nn {model} dataset metrics: f1micro:{metrics[3]}, f1macro:{metrics[2]}, accuracy:{metrics[1]}")

import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import joblib
import os 
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from sklearn.metrics import  precision_recall_fscore_support, classification_report

#load dataset 
os.chdir("..")
df = pd.read_csv("dataset/processed_datasets/ML_MED_Dataset_validation_Processed_onehot.csv")
X_test = df.iloc[:,0:29]
y_test = df.iloc[:,29:32]
df = pd.read_csv("dataset/processed_datasets/ML_MED_Dataset_validation_Processed_label.csv")
y_test_label = df.iloc[:,29:30]
y_test_label = np.ravel(y_test_label)

models = [""]
target_names = ['class 0', 'class 1', 'class 2']
for model in models:
    print(f"original_dataset")
    #randomforest metrics calculation
    print("Randomforest")
    rf = joblib.load(f"models/3classes/rf_model_best.sav")
    y_pred = rf.predict(X_test)
    class_prf = precision_recall_fscore_support(y_test_label, y_pred, average = None)
    print(classification_report(y_test_label, y_pred, target_names=target_names, zero_division=0.0))
    #adaboost metrics calculation
    print("Adaboost")
    ab = joblib.load(f"models/3classes/ab_model_best.sav") 
    y_pred = ab.predict(X_test)
    class_prf = precision_recall_fscore_support(y_test_label, y_pred, average = None)
    print(classification_report(y_test_label, y_pred, target_names=target_names, zero_division=0.0))
    #nn metrics calculation
    print("Neural_network")
    nn = load_model(f"models/3classes/trained_model_best.h5")
    y_pred = np.argmax(nn.predict(X_test, verbose=0), axis=1)
    print(classification_report(y_test_label, y_pred, target_names=target_names, zero_division=0.0))

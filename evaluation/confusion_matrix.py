import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import joblib
import seaborn as sn
import os 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import tensorflow_addons as tfa

#load dataset 
os.chdir("..")
df = pd.read_csv("dataset/processed_datasets/ML_MED_Dataset_test_Processed_onehot.csv")
X_test = df.iloc[:,0:29]
y_test = df.iloc[:,29:32]
df = pd.read_csv("dataset/processed_datasets/ML_MED_Dataset_test_Processed_label.csv")
y_test_label = df.iloc[:,29:30]
y_test_label = np.ravel(y_test_label)

models = ["optimal"]
#define classes 
classes = ["Digerente", "Endocrino", "Cardiovascolare"]

for model in models:
#randomforest metrics calculation ab_ddpm_model_best.sav
    rf = joblib.load(f"models/3classes/rf_{model}_model_best.sav")
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test_label, y_pred)
    print(cm)
    df_cfm = pd.DataFrame(cm ,index = classes, columns = classes)
    print(df_cfm)
    plt.figure(figsize = (12,10))
    ax = plt.axes()
    cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
    ax.set(xlabel = "Predicted label", ylabel = "True label")
    cfm_plot.figure.savefig(f"rf_{model}_confusion_matrix.png")
    #adaboost metrics calculation
    ab = joblib.load(f"models/3classes/ab_{model}_model_best.sav")
    y_pred = ab.predict(X_test)
    cm = confusion_matrix(y_test_label, y_pred)
    print(cm)
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (12,10))
    ax = plt.axes()
    cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
    ax.set(xlabel = "Predicted label", ylabel = "True label")
    cfm_plot.figure.savefig(f"ab_{model}_confusion_matrix.png")
    #nn metrics calculation
    nn = load_model(f"models/3classes/trained_model_{model}_best.h5")
    y_pred = (nn.predict(X_test)).argmax(axis = 1)
    cm = confusion_matrix(y_test_label, y_pred)
    print(cm)
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (12,10))
    ax = plt.axes()
    cfm_plot = sn.heatmap(df_cfm, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
    ax.set(xlabel = "Predicted label", ylabel = "True label")
    cfm_plot.figure.savefig(f"nn_{model}_confusion_matrix.png")


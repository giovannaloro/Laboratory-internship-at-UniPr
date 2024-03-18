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
df = pd.read_csv("dataset/ML_MED_Dataset_test_preprocessed_full.csv")
X_test = df.iloc[:,1:46]
y_test = df.iloc[:,46:52]
df = pd.read_csv("dataset/ML_MED_Dataset_test_preprocessed_full_label.csv")
y_test_label = df.iloc[:,46:47]
y_test_label = np.ravel(y_test_label)

#define classes 
classes = ["Apparato digerente", "Sistema endocrino", "Apparato urinario", "Sistema respiratorio", "Sistema cardiovascolare"]
#randomforest metrics calculation
rf = joblib.load("models/rf_model_smotenc_best.sav")
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test_label, y_pred)
df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize = (12,12))
ax = plt.axes()
cfm_plot = sn.heatmap(df_cfm, annot=True, ax=ax)
ax.set(xlabel = "Predicted label", ylabel = "True label")
cfm_plot.figure.savefig("rf_confusion_matrix.png")
#adaboost metrics calculation
ab = joblib.load("models/ab_model_smotenc_best.sav")
y_pred = ab.predict(X_test)
cm = confusion_matrix(y_test_label, y_pred)
df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize = (12,12))
ax = plt.axes()
cfm_plot = sn.heatmap(df_cfm, annot=True, ax=ax)
ax.set(xlabel = "Predicted label", ylabel = "True label")
cfm_plot.figure.savefig("ab_confusion_matrix.png")
#nn metrics calculation
nn = load_model("models/trained_model_smotenc_best.h5")
y_pred = (nn.predict(X_test)).argmax(axis = 1)
cm = confusion_matrix(y_test_label, y_pred)
df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize = (12,12))
ax = plt.axes()
cfm_plot = sn.heatmap(df_cfm, annot=True, ax=ax)
ax.set(xlabel = "Predicted label", ylabel = "True label")
cfm_plot.figure.savefig("nn_confusion_matrix.png")


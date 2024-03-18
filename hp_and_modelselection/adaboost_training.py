import pandas
import os 
import scipy 
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier

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
best_params = {'algorithm': 'SAMME', 'estimator': SGDClassifier(), 'learning_rate': 0.001, 'n_estimators': 150}
for i in range(100):
    ab = AdaBoostClassifier(**best_params)
    ab.fit(X_train, y_train)
    y_pred =ab.predict(X_validation)
    f_one_score = f1_score(y_validation, y_pred, average="macro")
    accuracy = accuracy_score(y_validation, y_pred)
    print(f"f1 score of model {i}:", f_one_score)
    filename = f"ab_model{i}.sav"
    joblib.dump(ab, filename)

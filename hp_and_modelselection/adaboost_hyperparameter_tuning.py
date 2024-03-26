import pandas
import os 
import scipy 
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier

#uplouading dataset 
os.chdir('..')
df = pandas.read_csv('dataset/processed_datasets/ML_MED_Dataset_train_Processed_label.csv')
X_train = df.iloc[:,0:29]
y_train = df.iloc[:,29:30]
y_train = np.ravel(y_train)
df = pandas.read_csv('dataset/processed_datasets/ML_MED_Dataset_validation_Processed_label.csv')
X_validation = df.iloc[:,0:29]
y_validation = df.iloc[:,29:30]
y_validation = np.ravel(y_validation)
#training and hyperparameters tuning
gnb = GaussianNB()
sgd = SGDClassifier()
dtc = DecisionTreeClassifier()
params = {'estimator':[gnb,sgd,dtc],'n_estimators':[150,200,250,300,350,400,450], 'algorithm':['SAMME'],'learning_rate':[0.001,0.001,0.01,0.1]}
adb = AdaBoostClassifier()
adb_tuned = GridSearchCV(estimator=adb, param_grid=params, n_jobs=-1, cv=8,verbose=3)
adb_tuned.fit(X_train, y_train)
print(adb_tuned.best_params_)
adb = AdaBoostClassifier(**adb_tuned.best_params_)
adb.fit(X_train,y_train)
y_pred = adb.predict(X_validation)
f_one_score = f1_score(y_validation, y_pred, average="weighted")
accuracy = accuracy_score(y_validation, y_pred)
print("f1 score:", f_one_score)
print("accuracy score:", accuracy)

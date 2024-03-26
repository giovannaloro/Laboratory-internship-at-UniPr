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


os.chdir('..')
datasets = []
for i in range(10):
    datasets.append(f"optimal_dataset_{i}.csv")
datasets_means = []
best_params = {'algorithm': 'SAMME', 'estimator': SGDClassifier(), 'learning_rate': 0.01, 'n_estimators': 300}
df = pandas.read_csv('dataset/processed_datasets/ML_MED_Dataset_validation_Processed_label.csv')
X_validation = df.iloc[:,0:29]
y_validation = df.iloc[:,29:30]
print(X_validation)
print(y_validation)
y_validation = np.ravel(y_validation)

for dataset in datasets:
    #load dataset 
    df = pandas.read_csv(f"{dataset}")
    X_train = df.iloc[:,0:29]
    y_train = df.iloc[:,29:30]
    print(X_train)
    print(y_train)
    y_train = np.ravel(y_train)
    df1 = len(df[df["tipo_operazione"]==0])
    df2 = len(df[df["tipo_operazione"]==1])
    df3 = len(df[df["tipo_operazione"]==2])
    print(dataset)
    print(f"class_0:{df1},class_1:{df2},class_2:{df3}")
    dataset_scores = []
    for i in range(10):
        ab = AdaBoostClassifier(**best_params)
        ab.fit(X_train, y_train)
        y_pred =ab.predict(X_validation)
        f_one_score = f1_score(y_validation, y_pred, average="macro")
        dataset_scores.append(f_one_score)
    datasets_means.append((np.mean(dataset_scores), dataset))
    dataset_scores = []

for mean in datasets_means:
    print(f"score of dataset {mean[1]} is: {mean[0]}")

#print(f"the best dataset is the number {np.argmax(datasets_means,axis=0)}")


    


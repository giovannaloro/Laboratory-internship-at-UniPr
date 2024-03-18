from sklearn.datasets import load_diabetes
from synthcity.plugins import Plugins
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader
import pandas as pd 
import math
import os 

os.chdir("..")
os.chdir("augmented_dataset") 
df = pd.read_csv("ML_MED_Dataset_smotenc_best.csv")
X_num = df.iloc[:,0:4]
X_cat = df.iloc[:,4:45]
y = df.iloc[:,45:46]
X_cat = X_cat.astype(int)
print(y)
X = pd.concat([X_num,X_cat], axis = 1)
print(X)
X["tipo_operazione"] = y
#X.to_csv("ax.csv", index=False)


models =  [ "tabulargan" ]

for model in models:
    for i in range(2):
        loader = GenericDataLoader(X, target_column="tipo_operazione")
        syn_model = Plugins().get(model)
        syn_model.fit(X)
        for j in range(5):
            generated = syn_model.generate(count=1000)
            print(generated)
            (generated.dataframe()).to_csv(f"dataset_{model}_smotenc_generated_{i}_{j}.csv", index=False)

from sklearn.datasets import load_diabetes
from synthcity.plugins import Plugins
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.constraints import Constraints
from synthcity.plugins.core.dataloader import GenericDataLoader
import pandas as pd 
import math
import random
import os 

os.chdir("..")
df = pd.read_csv("augmented_dataset/3classes/ML_MED_Dataset_train_Processed_smotenc_best_333.csv")
df = df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:,0:30]
print(X)


models =  [ "nflow","ddpm", "ctgan" ]

for model in models:
    for i in range(2):
        loader = GenericDataLoader(X, target_column="tipo_operazione")
        print(model)
        match model:
            case "ddpm":
                syn_model = Plugins().get(model, n_iter=100, batch_size=32, is_classification=True, model_type="mlp",model_params = dict(n_layers_hidden=10, n_units_hidden=256, dropout=0.0), random_state=random.randrange(0,100))
            case "ctgan":
                syn_model = Plugins().get(model, n_iter=100, batch_size=32, generator_n_layers_hidden = 5, generator_n_units_hidden = 256, discriminator_n_layers_hidden = 3, discriminator_n_units_hidden = 256, random_state=random.randrange(0,100))
            case "tvae":
                syn_model = Plugins().get(model, n_iter=100, batch_size=32, decoder_n_layers_hidden = 5, decoder_n_units_hidden=256, encoder_n_layers_hidden = 5, encoder_n_units_hidden = 256, random_state=random.randrange(0,100) )
            case "rtvae":
                syn_model = Plugins().get(model, n_iter=100, batch_size=32, decoder_n_layers_hidden = 5, decoder_n_units_hidden=256, encoder_n_layers_hidden = 5, encoder_n_units_hidden = 256, random_state=random.randrange(0,100) )
            case "nflow":
                syn_model = Plugins().get(model, n_iter=100, batch_size=32,n_layers_hidden = 10, n_units_hidden = 256, random_state=random.randrange(0,100))
        syn_model.fit(X)
        for j in range(5):
            dataset = syn_model.generate(count=1000).dataframe()
            dataset.to_csv(f"dataset_{model}_train_{i}_{j}.csv", index=False)

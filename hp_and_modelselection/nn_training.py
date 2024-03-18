import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras 
from tensorflow.keras.models import load_model
import numpy as np
import pandas 
import os 

#uplouading dataset 
os.chdir('..')
df = pandas.read_csv('augmented_dataset/ML_MED_Dataset_smotenc_best_onehot.csv')
X_train = df.iloc[:,0:45]
y_train = df.iloc[:,45:50]
print(X_train)
print(y_train)
df = pandas.read_csv('dataset/ML_MED_Dataset_validation_preprocessed_full.csv')
X_validation = df.iloc[:,1:46]
y_validation = df.iloc[:,46:51]
#training
model = load_model("models/tuned_noweight_model.keras")
print(model.get_config())
early_stop = keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0.03,
    patience=10,
    verbose=0,
    mode="min",
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=15
)
results_f1score_macro = []
f = open('results.txt', 'w')
f.close()

for run in range(100):
    model = load_model("models/tuned_noweight_model.keras")
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=10,
        epochs=100,
        callbacks=early_stop,
        shuffle=True,
        steps_per_epoch=None
        )
    run_result = model.evaluate(X_validation, y_validation)
    result = f"model number {run} metric: {model.metrics_names} : {run_result}"
    with open('results.txt', 'a') as f:
        f.write(f"{result}\n")
        f.close()
    model.save(f"trained_model_{run}.h5")
    results_f1score_macro.append(run_result[2])

macrof1_ordered_models = np.argsort(results_f1score_macro)
best_model = macrof1_ordered_models[len(macrof1_ordered_models)-1]
print(best_model)
with open('results.txt', 'a') as f:
    f.write(f"the best model is the number {best_model}")
    f.close()
print(f"the best model is the number {best_model}")

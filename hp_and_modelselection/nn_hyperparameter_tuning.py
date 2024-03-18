import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras 
import numpy as np
import keras_tuner
from tensorflow.keras.datasets import mnist
import pandas 
import os 

#importing dataset
os.chdir('..')
df = pandas.read_csv('dataset/ML_MED_Dataset_train_preprocessed_full.csv')
X_train = df.iloc[:,1:46]
y_train = df.iloc[:,46:52]
df = pandas.read_csv('dataset/ML_MED_Dataset_validation_preprocessed_full.csv')
X_validation = df.iloc[:,1:46]
y_validation = df.iloc[:,46:52]


#hyperparameter tuning 
def build_model(hp):

    model = keras.Sequential()

    input_dropout = hp.Choice("indrop", values=[0.1,0.2,0.3] )
    input_layer = keras.layers.Dense(46, input_shape=[46],activation='relu')
    model.add(keras.layers.Dropout(input_dropout))

    inner_layers=hp.Choice("inner_layers", values=[1,2,3])
    dense_dropout = hp.Choice("densedrop", values=[0.3,0.4,0.5,0.6])

    for x in range(inner_layers):
        model.add(keras.layers.Dense(46, activation='relu'))
        model.add(keras.layers.Dropout(dense_dropout))

    output_layer = keras.layers.Dense(5, activation='softmax')
    model.add(output_layer)

    learning_rate = hp.Choice("lr", values=[0.0001,0.001,0.01,0.1])
    momentum = hp.Choice("momentum", values=[0.0001,0.001,0.01,0.1])
    model.compile(optimizer=keras.optimizers.SGD( learning_rate=learning_rate, momentum=momentum), loss ='categorical_crossentropy', metrics=["categorical_accuracy",tfa.metrics.F1Score(average='macro',num_classes=5,name="macro_f1"),tfa.metrics.F1Score(average='micro',num_classes=5,name="micro_f1")])
    return model

tuner = keras_tuner.GridSearch(
    hypermodel=build_model,
    objective="loss",
    max_trials=20,
    executions_per_trial=5,
    overwrite=True,
    directory="hp_nn_search",
    project_name="Surgical_classification",
)

tuner.search(X_train, y_train, epochs=10, validation_data=(X_validation, y_validation))

# Get the top  hyperparameters.
best_hps = tuner.get_best_hyperparameters(6)
# Build the model with the best hp.
print(best_hps[0])
model = build_model(best_hps[0])
print(best_hps[0])
# Save the model.
model.save("tuned_noweight_model.keras")


import tensorflow as tf
import keras
import tensorflow_addons as tfa
from tensorflow import keras 
from tensorflow.keras.models import load_model
import numpy as np
import pandas 
import os 

#uplouading dataset 
os.chdir('..')
models=['optimal']
for model in models:
    df = pandas.read_csv(f'augmented_dataset/3classes/optimal_dataset_onehot.csv')
    X_train = df.iloc[:,0:29]
    y_train = df.iloc[:,29:32]
    df = pandas.read_csv('dataset/processed_datasets/ML_MED_Dataset_validation_Processed_onehot.csv')
    X_validation = df.iloc[:,0:29]
    y_validation = df.iloc[:,29:32]
    #training
    early_stop = keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0.03,
        patience=30,
        verbose=0,
        mode="min",
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=15
    )
    results_f1score_macro = []
    f = open(f'results_{model}.txt', 'w')
    f.close()

    for run in range(50):
        md = keras.Sequential()
        md.add(keras.layers.Dense(30, input_shape=[29],activation='relu'))
        md.add(keras.layers.Dropout(rate=0.2))
        md.add(keras.layers.Dense(30, input_shape=[29],activation='relu'))
        md.add(keras.layers.Dropout(rate=0.4))
        md.add(keras.layers.Dense(30, input_shape=[29],activation='relu'))
        md.add(keras.layers.Dropout(rate=0.4))
        md.add(keras.layers.Dense(3, activation='softmax'))
        md.compile(optimizer=keras.optimizers.SGD( learning_rate=0.001, momentum=0.01), loss ='categorical_crossentropy', metrics=["categorical_accuracy",tfa.metrics.F1Score(average='macro',num_classes=3,name="macro_f1"),tfa.metrics.F1Score(average='micro',num_classes=3,name="micro_f1")])
        md.fit(
            x=X_train,
            y=y_train,
            batch_size=16,
            epochs=100,
            callbacks=early_stop,
            shuffle=True,
            steps_per_epoch=None
            )
        run_result = md.evaluate(X_validation, y_validation)
        result = f"model number {run} metric: {md.metrics_names} : {run_result}"
        with open(f'results_{model}.txt', 'a') as f:
            f.write(f"{result}\n")
            f.close()
        md.save(f"trained_model_{model}_{run}.h5")
        results_f1score_macro.append(run_result[2])

    macrof1_ordered_models = np.argsort(results_f1score_macro)
    best_model = macrof1_ordered_models[len(macrof1_ordered_models)-1]
    print(best_model)
    with open(f'results_{model}.txt', 'a') as f:
        f.write(f"the best model  is the number {best_model}")
        f.close()
    print(f"the best {model} model is the number {best_model}")
